from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from experiments.mnist_analog_relu import runtime
from training.ibm_reram_hwa import IbmReramArrayPopulation


def _layout():
    return runtime.build_crossbar_layout((2, 2, 2), maximum_input_size=2)


def _population() -> IbmReramArrayPopulation:
    layout = _layout()
    size = sum(tile.cells for tile in layout)
    return IbmReramArrayPopulation(
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
        fingerprint="shadow-pv-runtime-fixture",
        aihwkit_version="1.1.0",
    )


def test_supervised_shadow_trains_every_coordinate_without_fault_or_teacher_input() -> None:
    layout = _layout()
    initial = torch.tensor(
        [0.50, 0.20, 0.30, 0.40, 0.20, -0.10, -0.20, 0.30],
        dtype=torch.float32,
    )
    train_loader = [
        (
            torch.tensor(
                [[1.0, 0.5], [0.25, 1.0], [0.8, 0.2], [0.2, 0.8]],
                dtype=torch.float32,
            ),
            torch.tensor([0, 1, 0, 1], dtype=torch.int64),
        )
    ]

    target, epochs, report, state = runtime._train_supervised_ce_shadow(
        initial_apparent_q=initial,
        logical_minimum=torch.full_like(initial, -1.0),
        logical_maximum=torch.full_like(initial, 1.0),
        layout=layout,
        digital_scales=(0.5, 0.25),
        train_loader=train_loader,
        repair_examples=4,
        epochs=1,
        maximum_batches=1,
        logical_learning_rates=(1e-3, 1e-3),
        betas=(0.9, 0.999),
        epsilon=1e-8,
        device=torch.device("cpu"),
    )

    signature = inspect.signature(runtime._train_supervised_ce_shadow)
    assert "teacher" not in signature.parameters
    assert "fault_mask" not in signature.parameters
    assert "plant" not in signature.parameters
    assert report["trainable_cells"] == initial.numel()
    assert report["frozen_cells"] == 0
    assert report["learner_accesses_fault_mask"] is False
    assert report["teacher_access_during_updates"] is False
    assert report["logical_learning_rates"] == pytest.approx([1e-3, 1e-3])
    assert report["effective_q_learning_rates"] == pytest.approx([2e-3, 4e-3])
    assert report["optimizer_steps"] == 1
    assert report["repair_cohort"]["examples"] == 4
    assert epochs[0]["train_cross_entropy"] > 0.0
    # Coordinate zero stands in for a physically stuck cell.  Because no mask
    # enters the learner, its digital target is optimized like every other q.
    assert target[0].item() != pytest.approx(initial[0].item())
    assert torch.all(target >= -1.0)
    assert torch.all(target <= 1.0)
    assert torch.equal(state["initial_post_fault_apparent_q"], initial)
    assert torch.equal(state["fixed_final_shadow_q"], target)


def test_shadow_program_verify_is_fault_blind_and_plant_keeps_stuck_q_immutable() -> None:
    layout = _layout()
    population = _population()
    plant = runtime.IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(7),
        device="cpu",
    )
    plant.persistent.zero_()
    plant.apparent.zero_()
    fault_mask = torch.zeros(plant.size, dtype=torch.bool)
    fault_mask[0] = True
    stuck = torch.zeros(plant.size)
    stuck[0] = 0.4
    plant.apply_stuck_at_fault_transition(
        mask=fault_mask,
        stuck_persistent_q=stuck,
        transition_id="test-shadow-pv-fault",
        source_population_fingerprint="published-shadow-pv-fixture",
    )
    faulted_state = plant.state_dict()
    target = torch.full((plant.size,), 0.25)
    target[0] = -0.5

    programming = runtime._program_shadow_target(
        controller_port=plant.controller_port(),
        shadow_target_q=target,
        maximum_programming_pulses=3,
        verify_tolerance_x=0.01,
    )
    final_state = plant.state_dict()
    effects = runtime._posthoc_recovery_pulse_effects(
        faulted_state=faulted_state,
        final_state=final_state,
        fault_mask=fault_mask,
        layout=layout,
    )

    signature = inspect.signature(runtime._program_shadow_target)
    assert "fault_mask" not in signature.parameters
    assert programming["eligible_mask_supplied"] is False
    assert programming["controller_accesses_fault_mask"] is False
    assert programming["commanded_pulses"] == effects["commanded_pulses"]
    assert programming["commanded_cells"] == plant.size
    assert programming["apparent_endpoint_x_sha256"] == runtime.tensor_sha256(
        (plant.apparent + 1.0) / 2.0
    )
    assert effects["commands_to_immutable_cells"] == 3
    assert effects["learner_accesses_fault_mask"] is False
    assert plant.persistent[0].item() == pytest.approx(0.4)
    assert torch.equal(
        final_state["persistent"][fault_mask],
        faulted_state["persistent"][fault_mask],
    )
    assert programming["budget_exhausted"] >= 1


def test_shadow_recovery_chronology_saves_target_and_binds_writer_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    layout = _layout()
    population = _population()
    plant = runtime.IbmOmEffectiveCrossbarPlant(
        population,
        generator=torch.Generator().manual_seed(13),
        device="cpu",
    )
    initial = torch.tensor(
        [0.50, 0.20, 0.30, 0.40, 0.20, -0.10, -0.20, 0.30],
        dtype=torch.float32,
    )
    plant.persistent.copy_(initial)
    plant.apparent.copy_(initial)
    train_inputs = torch.tensor(
        [[1.0, 0.5], [0.25, 1.0], [0.8, 0.2], [0.2, 0.8]],
        dtype=torch.float32,
    )
    train_labels = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    train_loader = [(train_inputs, train_labels)]
    predeployment_cohort, _ = runtime._ordered_labeled_cohort_report(
        inputs=[train_inputs],
        labels=[train_labels],
        split="predeployment_offchip_first_epoch_update_stream",
    )
    settings = SimpleNamespace(
        repair_examples=4,
        logical_learning_rates=(1e-3, 1e-3),
        maximum_programming_pulses=4,
        verify_tolerance_x=0.01,
        shadow_initial_state="post_fault_apparent_q",
        shadow_bounds="healthy_source_population_q_bounds",
        write_schedule="fixed_final_epoch_program_verify",
        writer="one_pulse_apparent_verify_persistent_handoff",
    )
    spec = SimpleNamespace(
        recovery=SimpleNamespace(
            policy="supervised_ce_shadow_program_verify",
            supervised_shadow_pv=settings,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            epochs=1,
            maximum_batches=1,
            layer_scope="all",
        ),
        runtime=SimpleNamespace(seed=42),
        device=SimpleNamespace(assignment_seed=87004),
        data=SimpleNamespace(num_points=4),
        evaluation=SimpleNamespace(
            maximum_validation_batches=1,
            sample_limit=None,
        ),
    )
    fault_mask = torch.zeros(plant.size, dtype=torch.bool)
    fault_mask[0] = True
    stuck = torch.zeros(plant.size)
    stuck[0] = 0.4
    context = {
        "faulted_checkpoint_path": tmp_path / "faulted.pt",
        "shadow_target_path": tmp_path / "shadow.pt",
        "fault_mask": fault_mask,
        "stuck_persistent_q": stuck,
        "fault_report": {"policy": "test"},
        "fault_source_population_fingerprint": "published-fixture",
        "predeployment_training_cohort": predeployment_cohort,
    }

    def fake_evaluate(*, effective_state: torch.Tensor, **_kwargs):
        return {
            "student_accuracy": float(effective_state.float().mean().item()),
            "examples": 4,
        }

    monkeypatch.setattr(runtime, "_evaluate", fake_evaluate)
    recovery, final_state = runtime._recover(
        plant=plant,
        spec=spec,
        endpoint_seed=89402,
        layout=layout,
        digital_scales=(0.5, 0.25),
        teacher=object(),
        validation_loader=train_loader,
        train_loader=train_loader,
        device=torch.device("cpu"),
        post_fault_context=context,
    )

    assert context["faulted_checkpoint_path"].is_file()
    assert context["shadow_target_path"].is_file()
    assert recovery["shadow_target_checkpoint_sha256"]
    assert recovery["optimizer"]["repair_cohort"][
        "same_as_predeployment_first_epoch_update_stream"
    ] is True
    assert recovery["program_verify"][
        "apparent_endpoint_matches_final_plant_apparent"
    ] is True
    assert recovery["program_verify"]["apparent_endpoint_q_sha256"] == (
        runtime.tensor_sha256(final_state["apparent"])
    )
    assert recovery["program_verify"]["post_program_persistent_q_sha256"] == (
        runtime.tensor_sha256(final_state["persistent"])
    )
    assert recovery["posthoc_shadow_fault_analysis"][
        "learner_accesses_fault_mask"
    ] is False
    assert torch.equal(
        final_state["persistent"][fault_mask],
        torch.load(
            context["faulted_checkpoint_path"],
            map_location="cpu",
            weights_only=True,
        )["persistent"][fault_mask],
    )
