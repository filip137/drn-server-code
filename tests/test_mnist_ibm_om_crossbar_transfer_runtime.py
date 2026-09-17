from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
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
    assert all(
        target.corruption_policy == "inherit_source"
        for target in spec.transfer.targets
    )


def test_transfer_target_can_override_source_corruption_policy() -> None:
    payload = _payload()
    payload["device"]["corruption_policy"] = "counterfactual_repaired"
    for target in payload["transfer"]["targets"]:
        target["corruption_policy"] = "published"

    spec = parse_crossbar_config(payload)

    assert spec.device.corruption_policy == "counterfactual_repaired"
    assert all(
        target.corruption_policy == "published"
        for target in spec.transfer.targets
    )
    assert runtime._resolve_transfer_target_corruption_policy(
        source_corruption_policy=spec.device.corruption_policy,
        target_corruption_policy=spec.transfer.targets[0].corruption_policy,
    ) == "published"


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ("published", "inherit_source", "published"),
        (
            "counterfactual_repaired",
            "inherit_source",
            "counterfactual_repaired",
        ),
        ("published", "counterfactual_repaired", "counterfactual_repaired"),
        ("counterfactual_repaired", "published", "published"),
    ],
)
def test_transfer_target_corruption_policy_resolution(
    source: str, target: str, expected: str
) -> None:
    assert runtime._resolve_transfer_target_corruption_policy(
        source_corruption_policy=source,
        target_corruption_policy=target,
    ) == expected


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
        (
            lambda transfer: transfer["targets"][0].__setitem__(
                "corruption_policy", "repair_if_needed"
            ),
            "inherit_source",
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
        ("support_clamped_no_update", [0.25, 0.4], "target_support_clamp"),
        ("continuous_hwa", [0.25, 0.4], "target_support_clamp"),
        ("stochastic_apparent_hwa", [0.25, 0.4], "target_support_clamp"),
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


def test_exact_master_handoff_is_fault_blind_on_every_target_population() -> None:
    offchip_state, source_plant, target_population, target_codebook = (
        _mapping_fixture()
    )

    mapped, report = runtime._map_transfer_source_state(
        source_state="offchip_fixed_final_master",
        offchip_policy="stochastic_apparent_hwa",
        offchip_state=offchip_state,
        source_plant=source_plant,
        target_population=target_population,
        target_codebook=target_codebook,
        deployment_target="fixed_final_master_fault_blind_pv",
    )

    assert torch.equal(mapped, offchip_state["fixed_final_master_q"])
    assert mapped.tolist() == pytest.approx([0.4, 0.4])
    assert report["target_mapping_rule"] == (
        "identity_fixed_final_master_fault_blind_pv_request"
    )
    assert report["target_codebook_index_sha256"] is None
    assert report["deployment_target"] == "fixed_final_master_fault_blind_pv"


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
        max_bound=torch.tensor([0.005, 1.0]),
        min_bound=torch.tensor([0.005, -1.0]),
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

    assert continuous[0].item() == pytest.approx(0.005)
    assert deterministic[0].item() == pytest.approx(0.005)
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
    assert plant.persistent[0].item() == pytest.approx(0.005)
    assert programming["trajectory_rng_backend"] == (
        "per_trajectory_stateless_counter_box_muller_v1"
    )
    assert "cpu_cuda_cross_backend_values_are_not_claimed_bit_exact" in (
        programming["trajectory_rng_reproducibility_scope"]
    )
    assert "62_bit_lane_pair_key_collision_space" in (
        programming["trajectory_rng_statistical_contract"]
    )
    assert len(programming["trajectory_seeds_sha256"]) == 64
    assert len(programming["trajectory_draw_indices_sha256"]) == 64
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


def test_matched_published_companion_defines_post_deployment_fault_overlay() -> None:
    published = _published_population()
    repaired = replace(
        published,
        corruption_policy="counterfactual_repaired",
        min_bound=torch.tensor([-1.0, -1.0]),
        max_bound=torch.tensor([1.0, 1.0]),
        dwmin_up=torch.tensor([0.1, 0.1]),
        dwmin_down=torch.tensor([0.1, 0.1]),
        corrupt=torch.zeros(2, dtype=torch.bool),
        fingerprint="repaired-runtime-fixture",
    )

    mask, stuck_persistent_q, report = runtime._matched_published_fault_overlay(
        healthy=repaired,
        published=published,
        preset_default_corrupt_devices_prob=0.0,
        enabled_corrupt_devices_prob=0.1348,
        corrupt_devices_range=0.01,
    )

    assert mask.tolist() == [True, False]
    assert stuck_persistent_q[0].item() == pytest.approx(0.005)
    assert report["published_corrupt_devices_probability"] == pytest.approx(0.1348)
    assert report["preset_default_corrupt_devices_probability"] == pytest.approx(0.0)
    assert report["corrupt_devices_range"] == pytest.approx(0.01)
    assert report["apparent_write_noise_retained"] is True
    assert report["faulted_cells"] == 1
    assert report["non_fault_identity_equal"] is True


def test_local_star_recovery_core_has_no_teacher_autograd_or_adam_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = runtime.build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    size = sum(tile.cells for tile in layout)
    population = IbmReramArrayPopulation(
        assignment_seed=87004,
        corruption_policy="counterfactual_repaired",
        binding_keys=tuple(tile.key for tile in layout),
        binding_shapes=tuple(tile.shape for tile in layout),
        binding_sampling_seeds=(11, 12),
        donor_sampling_seeds=(21, 22),
        nominal_dw_min=0.1,
        dw_min_std=0.0,
        write_noise_std=0.0,
        max_bound=torch.ones(size),
        min_bound=-torch.ones(size),
        dwmin_up=torch.full((size,), 0.1),
        dwmin_down=torch.full((size,), 0.1),
        reference=torch.zeros(size),
        corrupt=torch.zeros(size, dtype=torch.bool),
        published_corrupt=torch.zeros(size, dtype=torch.bool),
        fingerprint="star-local-runtime-fixture",
        aihwkit_version="1.1.0",
    )
    plant = runtime.IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(7),
        device="cpu",
    )
    fault_mask = torch.zeros(size, dtype=torch.bool)
    fault_mask[0] = True
    plant.apply_stuck_at_fault_transition(
        mask=fault_mask,
        stuck_persistent_q=torch.zeros(size),
        transition_id="test-fault",
        source_population_fingerprint="published-fixture",
    )
    spec = SimpleNamespace(
        recovery=SimpleNamespace(
            star=SimpleNamespace(
                pulse_rule="stochastic_pulse_sign_sgd",
                hidden_gain=1.0,
                output_gain=1.0,
            ),
            learning_rates_q=(0.1, 0.1),
            layer_scope="all",
            pulse_cap_per_cell=2,
            epochs=1,
            maximum_batches=1,
        ),
        runtime=SimpleNamespace(seed=42),
        device=SimpleNamespace(assignment_seed=87004),
    )
    targets = {
        "hidden_post_relu": torch.tensor([[0.5, 0.25]] * 10),
        "output_logits": torch.tensor([[0.75]] * 10),
    }
    bundle = SimpleNamespace(means=lambda name: targets[name].clone())
    monkeypatch.setattr(
        torch.autograd,
        "grad",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("autograd is forbidden in local STAR recovery")
        ),
    )
    monkeypatch.setattr(
        runtime,
        "PulseAdam",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("Adam is forbidden in local STAR recovery")
        ),
    )

    update_port = plant.local_star_update_port()
    assert not hasattr(update_port, "post_deployment_fault_mask")
    assert not hasattr(update_port, "post_deployment_stuck_persistent_q")
    assert not hasattr(update_port, "persistent")
    assert not hasattr(update_port, "population")
    epochs, optimizer, repair_identities = runtime._run_local_star_pulse_recovery(
        update_port=update_port,
        spec=spec,
        endpoint_seed=89402,
        layout=layout,
        digital_scales=(1.0, 1.0),
        target_bundle=bundle,
        train_loader=[
            (
                torch.tensor([[1.0, -0.5], [-0.25, 1.0]]),
                torch.tensor([0, 1]),
            )
        ],
        device=torch.device("cpu"),
    )

    assert epochs[0]["examples"] == 2
    assert optimizer["optimizer_steps"] == 2
    assert optimizer["repair_cohort"]["examples"] == 2
    assert len(repair_identities) == 2
    assert optimizer["digital_first_moment_values"] == 0
    assert optimizer["digital_second_moment_values"] == 0
    assert plant.persistent[0].item() == pytest.approx(0.0)


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


def test_local_star_state_error_report_is_teacher_free_and_label_addressed() -> None:
    layout = runtime.build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    size = sum(tile.cells for tile in layout)
    population = IbmReramArrayPopulation(
        assignment_seed=87004,
        corruption_policy="counterfactual_repaired",
        binding_keys=tuple(tile.key for tile in layout),
        binding_shapes=tuple(tile.shape for tile in layout),
        binding_sampling_seeds=(11, 12),
        donor_sampling_seeds=(21, 22),
        nominal_dw_min=0.1,
        dw_min_std=0.0,
        write_noise_std=0.0,
        max_bound=torch.ones(size),
        min_bound=-torch.ones(size),
        dwmin_up=torch.full((size,), 0.1),
        dwmin_down=torch.full((size,), 0.1),
        reference=torch.zeros(size),
        corrupt=torch.zeros(size, dtype=torch.bool),
        published_corrupt=torch.zeros(size, dtype=torch.bool),
        fingerprint="star-state-error-runtime-fixture",
        aihwkit_version="1.1.0",
    )
    plant = runtime.IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(7),
        device="cpu",
    )
    hidden_targets = torch.zeros((10, 2), dtype=torch.float32)
    hidden_targets[0] = torch.tensor([1.0, 2.0])
    hidden_targets[1] = torch.tensor([3.0, 4.0])
    output_targets = torch.zeros((10, 1), dtype=torch.float32)
    output_targets[0, 0] = 5.0
    output_targets[1, 0] = 6.0
    targets = {
        "hidden_post_relu": hidden_targets,
        "output_logits": output_targets,
    }
    bundle = SimpleNamespace(
        means=lambda name: targets[name].clone(),
        binding=SimpleNamespace(component_gains=(0.5, 0.25)),
        semantic_sha256="a" * 64,
    )

    report = runtime._evaluate_local_star_state_errors(
        plant=plant,
        target_bundle=bundle,
        digital_scales=(1.0, 1.0),
        layout=layout,
        loader=[
            (
                torch.tensor([[1.0, -0.5], [-0.25, 1.0]]),
                torch.tensor([0, 1]),
            )
        ],
        device=torch.device("cpu"),
        maximum_batches=1,
        sample_limit=None,
    )

    assert report["examples"] == 2
    assert report["teacher_access"] is False
    assert report["hidden_state_error_mean_squared_l2"] == pytest.approx(15.0)
    assert report["output_state_error_mean_squared_l2"] == pytest.approx(30.5)
    assert report["hidden_state_error_rms"] == pytest.approx((30.0 / 4.0) ** 0.5)
    assert report["output_state_error_rms"] == pytest.approx((61.0 / 2.0) ** 0.5)
    assert report["hidden_local_error_mean_squared_l2"] == pytest.approx(0.0)
    assert report["output_local_error_mean_squared_l2"] == pytest.approx(
        61.0 * 0.25**2 / 2.0
    )
    assert report["mean_local_objective"] == pytest.approx(7.5625)
    assert report["target_semantic_sha256"] == "a" * 64


def test_posthoc_star_audit_separates_commands_from_healthy_state_changes() -> None:
    layout = runtime.build_crossbar_layout((2, 2, 1), maximum_input_size=2)
    fault_mask = torch.tensor([True, False, False, True, False, False])
    initial_counts = torch.zeros(6, dtype=torch.int64)
    command_counts = torch.tensor([2, 1, 0, 3, 0, 1], dtype=torch.int64)
    before = torch.zeros(6, dtype=torch.float32)
    after = torch.tensor([0.0, 0.1, 0.0, 0.0, 0.0, -0.2])

    report = runtime._posthoc_local_star_pulse_effects(
        faulted_state={
            "persistent": before,
            "upward_pulses": initial_counts,
            "downward_pulses": initial_counts,
        },
        final_state={
            "persistent": after,
            "upward_pulses": command_counts,
            "downward_pulses": initial_counts,
        },
        fault_mask=fault_mask,
        layout=layout,
    )

    assert report["commanded_pulses"] == 7
    assert report["commands_to_immutable_cells"] == 5
    assert report["commands_to_programmable_cells"] == 2
    assert report["persistent_changed_healthy_cells"] == 2
    assert report["persistent_healthy_delta_l1"] == pytest.approx(0.3)
    assert report["per_layer"][0]["commanded_pulses"] == 6
    assert report["per_layer"][1]["commanded_pulses"] == 1


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


def test_supervised_ce_recovery_uses_labels_without_teacher_or_fault_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RestrictedPort:
        size = 1
        nominal_dw_min = 0.1

        def __init__(self) -> None:
            self._apparent = torch.tensor([0.25], dtype=torch.float32)

        @property
        def apparent(self) -> torch.Tensor:
            return self._apparent.clone()

        def state_hash_receipt(self) -> dict[str, str]:
            return {"apparent_sha256": "apparent", "persistent_sha256": "persistent"}

    gradients: list[torch.Tensor] = []

    class FakePulseAdam:
        def __init__(self, **_kwargs) -> None:
            pass

        def step(self, gradient: torch.Tensor, _port: RestrictedPort) -> dict:
            gradients.append(gradient.detach().clone())
            return {"commanded_pulses": 1, "applied_pulses": 1}

        def report(self) -> dict:
            return {
                "optimizer_steps": 1,
                "commanded_pulses": 1,
                "commanded_cells": 1,
            }

    def fake_logits(
        inputs: torch.Tensor,
        state: torch.Tensor,
        _layout,
        *,
        digital_scales,
    ) -> torch.Tensor:
        del digital_scales
        classes = torch.arange(10, dtype=state.dtype, device=state.device)
        return state[0] * classes.unsqueeze(0).expand(inputs.shape[0], -1)

    monkeypatch.setattr(runtime, "PulseAdam", FakePulseAdam)
    monkeypatch.setattr(runtime, "standard_crossbar_logits", fake_logits)
    spec = SimpleNamespace(
        recovery=SimpleNamespace(
            supervised_bp=SimpleNamespace(repair_examples=2),
            learning_rates_q=(6e-5, 6e-5),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            layer_scope="all",
            pulse_cap_per_cell=64,
            epochs=1,
            maximum_batches=1,
        ),
        runtime=SimpleNamespace(seed=42),
        device=SimpleNamespace(assignment_seed=87004),
    )
    inputs = torch.ones((2, 1), dtype=torch.float32)
    labels = torch.tensor([0, 9], dtype=torch.int64)
    expected_state = torch.tensor([0.25], requires_grad=True)
    expected_loss = torch.nn.functional.cross_entropy(
        fake_logits(inputs, expected_state, (), digital_scales=(1.0, 1.0)),
        labels,
    )
    expected_gradient = torch.autograd.grad(expected_loss, expected_state)[0]
    port = RestrictedPort()

    epochs, optimizer = runtime._run_supervised_ce_pulse_recovery(
        update_port=port,
        spec=spec,
        endpoint_seed=89402,
        layout=(),
        digital_scales=(1.0, 1.0),
        train_loader=[(inputs, labels)],
        device=torch.device("cpu"),
    )

    assert not hasattr(port, "post_deployment_fault_mask")
    assert not hasattr(port, "persistent")
    assert len(gradients) == 1
    assert torch.allclose(gradients[0], expected_gradient)
    assert epochs[0]["examples"] == 2
    assert epochs[0]["train_cross_entropy"] == pytest.approx(expected_loss.item())
    assert optimizer["repair_cohort"]["labels_sha256"]


def test_from_scratch_supervised_adam_does_not_require_an_offchip_cohort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cohort_fields = {
        "examples": 16,
        "ordered_sample_ids_sha256": "repair-samples",
        "model_inputs_sha256": "repair-inputs",
        "labels_sha256": "repair-labels",
        "ordered_example_identity_sequence_sha256": "repair-sequence",
        "unique_example_identities": 16,
    }

    class FakePlant:
        def __init__(self) -> None:
            self.apparent = torch.tensor([0.0])
            self.persistent = torch.tensor([0.0])
            self.population = SimpleNamespace(fingerprint="slow-population")

        def state_dict(self) -> dict:
            return {
                "apparent": self.apparent.clone(),
                "persistent": self.persistent.clone(),
            }

        def apply_stuck_at_fault_transition(self, **_kwargs) -> dict:
            return {"applied": True}

        def restricted_recovery_update_port(self) -> object:
            return object()

    evaluation = {
        "apparent_forward": {"student_accuracy": 0.1},
        "persistent_diagnostic": {"student_accuracy": 0.1},
    }
    monkeypatch.setattr(
        runtime,
        "_evaluate_plant_states",
        lambda **_kwargs: evaluation,
    )
    monkeypatch.setattr(
        runtime,
        "_run_supervised_ce_pulse_recovery",
        lambda **_kwargs: (
            [{"epoch": 1, "examples": 16}],
            {
                "repair_cohort": dict(cohort_fields),
                "commanded_pulses": 0,
                "commanded_cells": 0,
                "nominal_dw_min": 0.1,
            },
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_posthoc_recovery_pulse_effects",
        lambda **_kwargs: {"commanded_pulses": 0},
    )
    spec = SimpleNamespace(
        offchip=SimpleNamespace(policy="none"),
        data=SimpleNamespace(num_points=16),
        recovery=SimpleNamespace(
            policy="supervised_ce_pulse_adam",
            supervised_bp=SimpleNamespace(repair_examples=16),
            layer_scope="all",
            learning_rates_q=(6e-5, 6e-5),
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            epochs=1,
        ),
        device=SimpleNamespace(assignment_seed=87004),
        evaluation=SimpleNamespace(maximum_validation_batches=1, sample_limit=16),
    )
    predeployment = {
        "available": False,
        "reason": "no_predeployment_optimizer_updates",
        **{name: None for name in cohort_fields},
    }
    predeployment["examples"] = 0
    context = {
        "faulted_checkpoint_path": tmp_path / "faulted.pt",
        "fault_mask": torch.tensor([False]),
        "stuck_persistent_q": torch.tensor([0.0]),
        "fault_report": {},
        "fault_source_population_fingerprint": "fault-source",
        "predeployment_training_cohort": predeployment,
    }

    report, _ = runtime._recover(
        plant=FakePlant(),
        spec=spec,
        endpoint_seed=89402,
        layout=(),
        digital_scales=(1.0, 1.0),
        teacher=SimpleNamespace(),
        validation_loader=[],
        train_loader=[],
        device=torch.device("cpu"),
        post_fault_context=context,
    )

    assert report["optimizer"]["repair_cohort"][
        "same_as_predeployment_first_epoch_update_stream"
    ] is False
