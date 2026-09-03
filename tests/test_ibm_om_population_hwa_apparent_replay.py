from __future__ import annotations

import copy

import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_population_hwa_apparent_replay import (
    FORWARD_POLICY,
    prepare_apparent_endpoint,
    summarize_endpoints,
)
from experiments.mnist_relu_drn.ibm_om_population_hwa_cross_array_runtime import (
    DEPLOYMENT_SCHEMA,
    DEPLOYMENT_SCHEMA_VERSION,
)


def _deployment_payload() -> dict[str, object]:
    return {
        "schema": DEPLOYMENT_SCHEMA,
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "evidence_tier": "exploratory_noncanonical",
        "persistent_endpoint_applied_to_drn": True,
        "apparent_endpoint_applied_to_drn": False,
        "inference_read_noise": False,
        "target_raw_x": torch.tensor(
            [0.0, 0.1, 0.5, 0.9, 1.0, 0.6], dtype=torch.float32
        ),
        "programming": {
            "accepted": torch.tensor(
                [True, True, False, True, True, False], dtype=torch.bool
            )
        },
        "continuation_state": {
            "state_coordinate": "native_raw_active_a",
            "persistent": torch.tensor(
                [-1.0, -0.8, 0.0, 0.8, 1.0, 0.2], dtype=torch.float32
            ),
            "apparent": torch.tensor(
                [-1.4, -1.0, 0.0, 1.0, 1.4, 0.2], dtype=torch.float32
            ),
        },
    }


def test_prepare_apparent_endpoint_projects_to_positive_g_without_mutation() -> None:
    payload = _deployment_payload()
    original = copy.deepcopy(payload)

    persistent_g, apparent_g, report = prepare_apparent_endpoint(
        payload,
        binding_shapes=((2, 2), (1, 2)),
    )

    torch.testing.assert_close(
        persistent_g,
        torch.tensor([0.0, 0.2, 1.0, 1.8, 2.0, 1.2]),
    )
    torch.testing.assert_close(
        apparent_g,
        torch.tensor([0.0, 0.0, 1.0, 2.0, 2.0, 1.2]),
    )
    assert report["policy"] == FORWARD_POLICY
    assert report["below_zero"] == 1
    assert report["above_one"] == 1
    assert report["out_of_range"] == 2
    assert report["out_of_range_fraction"] == pytest.approx(2.0 / 6.0)
    assert report["accepted_out_of_range"] == 2
    assert report["layers"]["0"]["out_of_range"] == 1
    assert report["layers"]["1"]["out_of_range"] == 1
    assert report["literal_apparent_used_as_unbounded_network_conductance"] is False
    assert report["projection_is_diagnostic_not_additional_programming"] is True
    assert torch.equal(
        payload["continuation_state"]["persistent"],
        original["continuation_state"]["persistent"],
    )
    assert torch.equal(
        payload["continuation_state"]["apparent"],
        original["continuation_state"]["apparent"],
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("persistent_endpoint_applied_to_drn", False),
        ("apparent_endpoint_applied_to_drn", True),
        ("inference_read_noise", True),
    ),
)
def test_prepare_apparent_endpoint_rejects_changed_source_semantics(
    field: str,
    value: bool,
) -> None:
    payload = _deployment_payload()
    payload[field] = value

    with pytest.raises(ValueError, match="semantic contract"):
        prepare_apparent_endpoint(payload, binding_shapes=((2, 2), (1, 2)))


def _endpoint_record(
    *,
    state: str,
    population: str,
    apparent_accuracy: float,
    persistent_accuracy: float,
) -> dict[str, object]:
    return {
        "array_label": "B",
        "state_label": state,
        "population_role": population,
        "apparent_projected_test": {"student_accuracy": apparent_accuracy},
        "persistent_comparison_test": {
            "student_accuracy": persistent_accuracy
        },
        "state_diagnostics": {
            "cells": 10,
            "out_of_range": 2,
            "accepted_out_of_range": 1,
        },
    }


def test_summarize_endpoints_keeps_hwa_and_corruption_axes_separate() -> None:
    values = {
        "raw_relu": {
            "counterfactual_repaired": 0.90,
            "published_corrupt": 0.70,
        },
        "population_hwa_continuous": {
            "counterfactual_repaired": 0.92,
            "published_corrupt": 0.80,
        },
        "population_hwa_one_delta_qat": {
            "counterfactual_repaired": 0.91,
            "published_corrupt": 0.79,
        },
    }
    records = [
        _endpoint_record(
            state=state,
            population=population,
            apparent_accuracy=accuracy,
            persistent_accuracy=accuracy - 0.01,
        )
        for state, populations in values.items()
        for population, accuracy in populations.items()
    ]

    summary = summarize_endpoints(records)

    array_b = summary["arrays"]["B"]
    assert array_b["states"]["raw_relu"][
        "published_corruption_penalty_accuracy"
    ] == pytest.approx(0.20)
    assert array_b["states"]["population_hwa_continuous"][
        "published_corruption_penalty_accuracy"
    ] == pytest.approx(0.12)
    assert array_b["comparisons"]["counterfactual_repaired"] == pytest.approx(
        {
            "continuous_hwa_minus_raw_relu": 0.02,
            "one_delta_qat_hwa_minus_raw_relu": 0.01,
            "one_delta_qat_minus_continuous_hwa": -0.01,
        }
    )
    assert array_b["comparisons"]["published_corrupt"] == pytest.approx(
        {
            "continuous_hwa_minus_raw_relu": 0.10,
            "one_delta_qat_hwa_minus_raw_relu": 0.09,
            "one_delta_qat_minus_continuous_hwa": -0.01,
        }
    )
    assert summary["transfer_B_D"]["raw_relu"][
        "published_corruption_penalty_accuracy"
    ] == pytest.approx(0.20)
