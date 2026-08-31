from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.mnist_analog_relu.config import parse_crossbar_config
from experiments.mnist_analog_relu import runtime
from experiments.schema import ConfigError
from training.ibm_reram_hwa import IbmReramArrayPopulation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "examples"
    / "mnist_analog_relu"
    / "ibm_om_onchip_importance"
    / "smoke_matched_winsorized_frozen.json"
)


def _payload() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_transfer_config_parses_explicit_source_and_multiple_fresh_arrays() -> None:
    spec = parse_crossbar_config(_payload())

    assert spec.transfer.enabled is True
    assert spec.transfer.source_state == "offchip_fixed_final_master"
    assert [target.assignment_seed for target in spec.transfer.targets] == [
        87005,
        87006,
        87007,
    ]
    assert all(
        len(target.endpoint_seeds) == len(spec.device.endpoint_seeds)
        for target in spec.transfer.targets
    )


def test_disabled_transfer_requires_none_source_and_no_targets() -> None:
    payload = _payload()
    payload["transfer"] = {
        "enabled": False,
        "source_state": "none",
        "targets": [],
    }

    spec = parse_crossbar_config(payload)

    assert spec.transfer.enabled is False
    assert spec.transfer.source_state == "none"
    assert spec.transfer.targets == ()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda transfer: transfer["targets"][1].__setitem__(
                "assignment_seed", transfer["targets"][0]["assignment_seed"]
            ),
            "unique target assignment seeds",
        ),
        (
            lambda transfer: transfer["targets"][0].__setitem__(
                "assignment_seed", 87004
            ),
            "differ from the source assignment",
        ),
        (
            lambda transfer: transfer["targets"][0].__setitem__(
                "endpoint_seeds", [89502]
            ),
            "one target endpoint per source endpoint",
        ),
        (
            lambda transfer: transfer.__setitem__("source_state", "apparent"),
            "offchip_fixed_final_master",
        ),
    ],
)
def test_transfer_config_rejects_ambiguous_or_unmatched_targets(
    mutation,
    message: str,
) -> None:
    payload = deepcopy(_payload())
    mutation(payload["transfer"])

    with pytest.raises(ConfigError, match=message):
        parse_crossbar_config(payload)


def _mapping_fixture() -> tuple[dict, SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    offchip_state = {
        "fixed_final_master_q": torch.tensor([0.4, 0.4], dtype=torch.float32),
        "fixed_final_realized_q": torch.tensor([-0.3, -0.3], dtype=torch.float32),
    }
    source_plant = SimpleNamespace(
        persistent=torch.tensor([0.8, -0.8], dtype=torch.float32),
        apparent=torch.tensor([-0.9, 0.9], dtype=torch.float32),
    )
    target_population = SimpleNamespace(
        logical_min=torch.tensor([-0.25, -0.5], dtype=torch.float32),
        logical_max=torch.tensor([0.25, 0.5], dtype=torch.float32),
    )
    target_codebook = SimpleNamespace(
        values=torch.tensor(
            [
                [-0.25, -0.5],
                [0.0, 0.0],
                [0.25, 0.5],
            ],
            dtype=torch.float32,
        )
    )
    return offchip_state, source_plant, target_population, target_codebook


@pytest.mark.parametrize(
    ("policy", "expected", "rule"),
    [
        ("none", [0.4, 0.4], "identity_master_q"),
        ("continuous_hwa", [0.25, 0.4], "target_support_clamp"),
        (
            "deterministic_qat",
            [0.25, 0.5],
            "target_deterministic_codebook_projection",
        ),
    ],
)
def test_offchip_transfer_maps_master_separately_on_target_population(
    policy: str,
    expected: list[float],
    rule: str,
) -> None:
    offchip_state, source_plant, target_population, target_codebook = (
        _mapping_fixture()
    )

    mapped, report = runtime._map_transfer_source_state(
        source_state="offchip_fixed_final_master",
        offchip_policy=policy,
        offchip_state=offchip_state,
        source_plant=source_plant,
        target_population=target_population,
        target_codebook=target_codebook,
    )

    assert mapped.tolist() == pytest.approx(expected)
    assert report["target_mapping_rule"] == rule
    assert report["source_q_sha256"] == runtime.tensor_sha256(
        offchip_state["fixed_final_master_q"]
    )
    assert report["copied_source_offchip_realized_or_codebook_state"] is False
    assert not torch.equal(mapped, offchip_state["fixed_final_realized_q"])
    assert not torch.equal(mapped, source_plant.persistent)


def test_same_array_transfer_pairs_hidden_persistent_state_not_apparent() -> None:
    offchip_state, source_plant, target_population, target_codebook = (
        _mapping_fixture()
    )

    mapped, report = runtime._map_transfer_source_state(
        source_state="same_array_persistent",
        offchip_policy="deterministic_qat",
        offchip_state=offchip_state,
        source_plant=source_plant,
        target_population=target_population,
        target_codebook=target_codebook,
    )

    assert torch.equal(mapped, source_plant.persistent)
    assert not torch.equal(mapped, source_plant.apparent)
    assert report["target_mapping_rule"] == (
        "identity_source_final_hidden_persistent_q"
    )
    assert report["copied_source_persistent_state"] is True


def _published_population() -> IbmReramArrayPopulation:
    corrupt = torch.tensor([True, False])
    return IbmReramArrayPopulation(
        assignment_seed=87004,
        corruption_policy="published",
        binding_keys=("crossbar.layer0.tile0", "crossbar.layer1.tile0"),
        binding_shapes=((1, 1), (1, 1)),
        binding_sampling_seeds=(11, 12),
        donor_sampling_seeds=(21, 22),
        nominal_dw_min=0.1,
        dw_min_std=0.0,
        write_noise_std=0.0,
        max_bound=torch.tensor([0.05, 1.0]),
        min_bound=torch.tensor([0.05, -1.0]),
        dwmin_up=torch.tensor([0.0, 0.1]),
        dwmin_down=torch.tensor([0.0, 0.1]),
        reference=torch.zeros(2),
        corrupt=corrupt,
        published_corrupt=corrupt.clone(),
        fingerprint="published-runtime-fixture",
        aihwkit_version="1.1.0",
    )


def test_published_defects_are_explicit_in_mapping_and_programming_reports() -> None:
    population = _published_population()
    codebook = runtime.build_deterministic_effective_codebook(
        population,
        maximum_pulses=3,
    )
    requested = torch.tensor([0.8, 0.8])
    continuous = torch.maximum(
        torch.minimum(requested, population.logical_max),
        population.logical_min,
    )
    deterministic, indices = runtime.project_to_nearest_effective_code(
        codebook,
        requested,
    )
    mapping = runtime._mapping_report(
        population=population,
        requested=requested,
        continuous=continuous,
        codebook=deterministic,
        pulse_indices=indices,
        level_counts=codebook.effective_level_counts,
    )

    assert continuous[0].item() == pytest.approx(0.05)
    assert deterministic[0].item() == pytest.approx(0.05)
    assert mapping["defects"]["final_corrupt_cells"] == 1
    assert mapping["defects"]["published_corrupt_cells"] == 1
    assert mapping["final_corrupt_one_level_cells"] == 1
    assert mapping["defects"]["per_binding"] == [
        {
            "binding_key": "crossbar.layer0.tile0",
            "binding_shape": [1, 1],
            "cell_offset_start": 0,
            "cell_offset_stop": 1,
            "cells": 1,
            "final_corrupt_cells": 1,
            "published_corrupt_cells": 1,
            "repaired_published_corrupt_cells": 0,
            "unattributed_final_corrupt_cells": 0,
            "collapsed_zero_step_cells": 1,
        },
        {
            "binding_key": "crossbar.layer1.tile0",
            "binding_shape": [1, 1],
            "cell_offset_start": 1,
            "cell_offset_stop": 2,
            "cells": 1,
            "final_corrupt_cells": 0,
            "published_corrupt_cells": 0,
            "repaired_published_corrupt_cells": 0,
            "unattributed_final_corrupt_cells": 0,
            "collapsed_zero_step_cells": 0,
        },
    ]

    plant, programming = runtime._program_endpoint(
        population=population,
        requested=requested,
        assignment_seed=87004,
        endpoint_seed=89402,
        maximum_pulses=3,
        tolerance_x=0.01,
        device=torch.device("cpu"),
        stream_role="published_defect_test",
        random_stream_fingerprint=population.fingerprint,
    )
    assert plant.persistent[0].item() == pytest.approx(0.05)
    assert programming["defects"]["final_corrupt_cells"] == 1
    assert programming["defects"]["published_corrupt_cells"] == 1
    assert programming["plant_pulses"]["pulses_to_final_corrupt_cells"] == 3
    assert programming["final_corrupt_endpoint"] == {
        "cells": 1,
        "exact_target_in_support": 0,
        "exact_target_outside_support": 1,
        "verify_window_intersects_support": 0,
        "verify_window_disjoint_support": 1,
        "apparent_accepted": 0,
        "persistent_within_tolerance": 0,
        "apparent_accepted_persistent_outside_tolerance": 0,
        "budget_exhausted": 1,
        "nonfinite": 0,
        "verify_reads": 4,
        "programming_pulses": 3,
        "final_saturated_lower": 1,
        "final_saturated_upper": 1,
    }


def test_plant_evaluation_reports_apparent_forward_and_persistent_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plant = SimpleNamespace(
        apparent=torch.tensor([0.75], dtype=torch.float32),
        persistent=torch.tensor([-0.25], dtype=torch.float32),
    )
    seen: list[torch.Tensor] = []

    def fake_evaluate(*, effective_state: torch.Tensor, **_kwargs) -> dict:
        seen.append(effective_state.detach().clone())
        return {"student_accuracy": float(effective_state[0].item())}

    monkeypatch.setattr(runtime, "_evaluate", fake_evaluate)

    report = runtime._evaluate_plant_states(
        plant=plant,
        digital_scales=(1.0, 1.0),
        layout=(),
        teacher=object(),
        loader=[],
        device=torch.device("cpu"),
        maximum_batches=None,
        sample_limit=None,
    )

    assert report["network_forward_state"] == "apparent_q"
    assert report["hidden_update_state"] == "persistent_q"
    assert torch.equal(seen[0], plant.apparent)
    assert torch.equal(seen[1], plant.persistent)
    assert report["apparent_forward"]["student_accuracy"] == pytest.approx(0.75)
    assert report["persistent_diagnostic"]["student_accuracy"] == pytest.approx(
        -0.25
    )


def test_recovery_gradient_uses_apparent_forward_with_persistent_pulse_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePlant:
        def __init__(self) -> None:
            self.apparent = torch.tensor([0.75], dtype=torch.float32)
            self.persistent = torch.tensor([-0.25], dtype=torch.float32)
            self.population = SimpleNamespace(nominal_dw_min=0.1)
            self.size = 1

        def state_dict(self) -> dict:
            return {
                "apparent": self.apparent.clone(),
                "persistent": self.persistent.clone(),
            }

        def load_state_dict(self, state: dict) -> None:
            self.apparent.copy_(state["apparent"])
            self.persistent.copy_(state["persistent"])

    forward_states: list[torch.Tensor] = []
    optimizer_gradients: list[torch.Tensor] = []

    def fake_logits(
        inputs: torch.Tensor,
        state: torch.Tensor,
        _layout,
        *,
        digital_scales,
    ) -> torch.Tensor:
        del digital_scales
        forward_states.append(state.detach().clone())
        classes = torch.arange(10, dtype=state.dtype, device=state.device)
        return state[0] * classes.unsqueeze(0).expand(inputs.shape[0], -1)

    def fake_evaluate_states(*, plant, **_kwargs) -> dict:
        return {
            "network_forward_state": "apparent_q",
            "hidden_update_state": "persistent_q",
            "apparent_forward": {
                "student_accuracy": float(plant.apparent[0].item())
            },
            "persistent_diagnostic": {
                "student_accuracy": float(plant.persistent[0].item())
            },
        }

    class FakePulseAdam:
        def __init__(self, **_kwargs) -> None:
            pass

        def step(self, gradient: torch.Tensor, plant: FakePlant) -> dict:
            optimizer_gradients.append(gradient.detach().clone())
            plant.persistent.add_(0.1)
            plant.apparent.add_(0.2)
            return {"applied_pulses": 1}

        def report(self) -> dict:
            return {"applied_pulses": 1}

    monkeypatch.setattr(runtime, "standard_crossbar_logits", fake_logits)
    monkeypatch.setattr(runtime, "_evaluate_plant_states", fake_evaluate_states)
    monkeypatch.setattr(runtime, "PulseAdam", FakePulseAdam)
    plant = FakePlant()
    spec = SimpleNamespace(
        recovery=SimpleNamespace(
            policy="pulse_adam",
            layer_scope="all",
            learning_rates_q=(1e-4, 1e-4),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            pulse_cap_per_cell=1,
            epochs=1,
            maximum_batches=1,
        ),
        runtime=SimpleNamespace(seed=42),
        device=SimpleNamespace(assignment_seed=87004),
        evaluation=SimpleNamespace(
            maximum_validation_batches=1,
            sample_limit=2,
        ),
    )
    teacher = SimpleNamespace(
        logits=lambda inputs: torch.zeros(
            (inputs.shape[0], 10), dtype=torch.float32
        )
    )
    batch = (torch.ones((2, 1), dtype=torch.float32), torch.zeros(2))

    report, _state = runtime._recover(
        plant=plant,
        spec=spec,
        endpoint_seed=89402,
        layout=(),
        digital_scales=(1.0, 1.0),
        teacher=teacher,
        validation_loader=[batch],
        train_loader=[batch],
        device=torch.device("cpu"),
    )

    assert torch.equal(forward_states[0], torch.tensor([0.75]))
    assert not torch.equal(forward_states[0], torch.tensor([-0.25]))
    assert len(optimizer_gradients) == 1
    assert report["network_forward_state"] == "apparent_q"
    assert report["hidden_update_state"] == "persistent_q"
    assert report["gradient_handoff"] == (
        "identity_ste_apparent_q_to_persistent_pulse_update"
    )
