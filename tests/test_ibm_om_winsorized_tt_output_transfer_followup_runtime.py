from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.mnist_relu_drn import (
    ibm_om_winsorized_endpoint_optimizer_recovery_runtime as base_runtime,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_tt_output_transfer_followup_config import (
    FROZEN_SELECTED_LAMBDAS,
    parse_tt_output_transfer_followup_config,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_tt_output_transfer_followup_runtime import (
    _candidate_effective_lambdas,
    _with_output_layer_diagnostic,
)


def _protocol():
    return parse_tt_output_transfer_followup_config(
        {
            "schema_version": 1,
            "experiment_id": (
                "mnist_ibm_om_winsorized_tt_output_transfer_followup.v1"
            ),
            "target_assignment_seed": 87004,
            "development_endpoint_seed": 89401,
            "evaluation_endpoint_seeds": [89402, 89403, 89404, 89405],
            "w2_multiplier_grid": [10.0, 30.0, 100.0],
        }
    ).protocol


def test_w2_grid_preserves_w1_and_changes_only_selected_w2_lambda() -> None:
    protocol = _protocol()
    for variant in ("tt_v1", "tt_v2"):
        source_w1, source_w2 = FROZEN_SELECTED_LAMBDAS[variant]
        for multiplier in (10.0, 30.0, 100.0):
            candidate = _candidate_effective_lambdas(
                protocol, variant=variant, w2_multiplier=multiplier
            )
            assert candidate[0] == source_w1
            assert candidate[1] == source_w2 * multiplier
    with pytest.raises(ValueError, match="outside the frozen development grid"):
        _candidate_effective_lambdas(
            protocol, variant="tt_v1", w2_multiplier=3.0
        )


def test_output_layer_diagnostic_exposes_c_crossing_and_cap_fields() -> None:
    source = FROZEN_SELECTED_LAMBDAS["tt_v2"]
    multiplier = 30.0
    report = {
        "hyperparameters": {
            "effective_transfer_lambdas": [source[0], source[1] * multiplier]
        },
        "slow_c_pulse": {
            "by_layer": [700, 23],
            "cells_at_cap_by_layer": [4, 5],
            "maximum_per_cell_by_layer": [64, 63],
        },
        "h": {
            "threshold_crossings_by_layer": [700, 29],
            "cap_blocked_debt_retained_by_layer_across_steps": [1, 6],
            "final_pulse_unit_state_by_layer": [
                {"absolute_maximum": 0.5},
                {"absolute_maximum": 3.25},
            ],
        },
    }
    enriched = _with_output_layer_diagnostic(
        report,
        variant="tt_v2",
        w2_multiplier=multiplier,
        source_lambdas=source,
    )
    output = enriched["output_layer_transfer"]
    assert output["w1_effective_lambda_frozen"] == source[0]
    assert output["w2_effective_lambda"] == source[1] * multiplier
    assert output["w2_slow_c_pulses"] == 23
    assert output["w2_slow_c_cells_at_cap"] == 5
    assert output["w2_slow_c_maximum_pulses_per_cell"] == 63
    assert output["w2_h_threshold_crossings"] == 29
    assert output["w2_h_cap_blocked_debt_retained_across_steps"] == 6
    assert output["w2_final_h_pulse_unit_state"]["absolute_maximum"] == 3.25


def test_run_arm_routes_explicit_tt_lambdas_only_to_optimizer_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StopAfterBuild(RuntimeError):
        pass

    seen: dict[str, object] = {}

    def restore(endpoint, *, pulse_population, protocol, device):
        seen["restore"] = {
            "endpoint": endpoint,
            "pulse_population": pulse_population,
            "protocol": protocol,
            "device": device,
        }
        return SimpleNamespace(persistent=torch.zeros(2))

    def build_optimizer(**kwargs):
        seen["build"] = kwargs
        raise StopAfterBuild

    monkeypatch.setattr(base_runtime, "_restore_c", restore)
    monkeypatch.setattr(base_runtime, "_build_optimizer", build_optimizer)
    explicit = (0.25, 0.75)
    with pytest.raises(StopAfterBuild):
        base_runtime._run_arm(
            phase="routing_regression",
            optimizer_name="tt_v1",
            candidate_scale=10.0,
            maximum_batches=1,
            endpoint_path=Path("p0.pt"),
            endpoint={"p0": True},
            endpoint_seed=89401,
            population=object(),
            pulse_population=object(),
            symmetry=object(),
            stack=SimpleNamespace(device=torch.device("cpu")),
            teacher=object(),
            loaders=object(),
            train_generator_state=torch.zeros(1, dtype=torch.uint8),
            protocol=object(),
            store=object(),
            before_validation={},
            before_test=None,
            tt_effective_transfer_lambdas=explicit,
        )
    assert "tt_effective_transfer_lambdas" not in inspect.signature(
        base_runtime._restore_c
    ).parameters
    assert seen["build"]["tt_effective_transfer_lambdas"] == explicit
