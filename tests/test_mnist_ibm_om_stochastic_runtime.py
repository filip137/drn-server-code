from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from experiments.mnist_analog_relu import runtime
from training.ibm_om_standard_crossbar import (
    IbmOmEffectiveCrossbarPlant,
    flatten_logical_crossbar_matrices,
)
from training.ibm_reram_hwa import IbmReramArrayPopulation


def _layout():
    return runtime.build_crossbar_layout((2, 2, 2), maximum_input_size=2)


def _population(
    *,
    assignment_seed: int,
    corruption_policy: str = "counterfactual_repaired",
) -> IbmReramArrayPopulation:
    layout = _layout()
    size = sum(tile.cells for tile in layout)
    corrupt = torch.zeros(size, dtype=torch.bool)
    minimum = torch.full((size,), -1.0)
    maximum = torch.full((size,), 1.0)
    upward = torch.full((size,), 0.1)
    downward = torch.full((size,), 0.1)
    if corruption_policy == "published":
        corrupt[0] = True
        minimum[0] = 0.2
        maximum[0] = 0.2
        upward[0] = 0.0
        downward[0] = 0.0
    return IbmReramArrayPopulation(
        assignment_seed=assignment_seed,
        corruption_policy=corruption_policy,
        binding_keys=tuple(tile.key for tile in layout),
        binding_shapes=tuple(tile.shape for tile in layout),
        binding_sampling_seeds=(11, 12),
        donor_sampling_seeds=(21, 22),
        nominal_dw_min=0.1,
        dw_min_std=0.0,
        write_noise_std=0.0,
        max_bound=maximum,
        min_bound=minimum,
        dwmin_up=upward,
        dwmin_down=downward,
        reference=torch.zeros(size),
        corrupt=corrupt,
        published_corrupt=corrupt.clone(),
        fingerprint=f"stochastic-runtime-{assignment_seed}-{corruption_policy}",
        aihwkit_version="1.1.0",
    )


def _plant() -> IbmOmEffectiveCrossbarPlant:
    layout = _layout()
    plant = IbmOmEffectiveCrossbarPlant(
        _population(assignment_seed=87004),
        generator=torch.Generator().manual_seed(17),
        device="cpu",
    )
    q = flatten_logical_crossbar_matrices(
        (
            torch.tensor([[0.5, -0.3], [0.2, 0.4]], dtype=torch.float32),
            torch.tensor([[0.3, -0.2], [-0.1, 0.25]], dtype=torch.float32),
        ),
        layout,
    )
    plant.persistent.copy_(q)
    plant.apparent.copy_(q)
    return plant


def _batch():
    return (
        torch.tensor(
            [[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.2, 0.8]],
            dtype=torch.float32,
        ),
        torch.tensor([0, 1, 0, 1], dtype=torch.int64),
    )


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        repair_examples=4,
        label_source="ground_truth",
        update_batching="minibatch",
        gradient_engine="manual_cross_entropy_backprop",
        optimizer_state="none",
        weight_state="physical_persistent_and_apparent_device_state_no_shadow",
        pulse_type="stochastic_compressed",
        desired_bl=31,
        fixed_bl=True,
        update_bl_management=True,
        update_management=True,
        um_grad_scale=1.0,
        bit_line_seed=108402,
        cumulative_pulse_cap=None,
        final_program_verify=False,
    )


def _spec(policy: str) -> SimpleNamespace:
    tiki_taka = None
    if policy == "supervised_ce_tiki_taka_v1":
        tiki_taka = SimpleNamespace(
            algorithm="tiki_taka_transfer_compound_v1",
            gamma=0.0,
            fast_lr=1.0,
            transfer_every=1,
            units_in_mbatch=True,
            n_reads_per_transfer=1,
            transfer_selection="sequential_physical_tile_columns",
            transfer_lr=1.0,
            scale_transfer_lr=True,
            transfer_columns=True,
            with_reset_prob=0.0,
            random_selection=False,
            fast_assignment_seed=88004,
            fast_endpoint_seeds=(98402,),
            fast_corruption_policy="published",
        )
    return SimpleNamespace(
        recovery=SimpleNamespace(
            policy=policy,
            supervised_stochastic_bp=_settings(),
            tiki_taka=tiki_taka,
            learning_rates_q=(1.0, 1.0),
            epochs=1,
            maximum_batches=1,
            layer_scope="all",
        ),
        runtime=SimpleNamespace(seed=42),
        device=SimpleNamespace(
            assignment_seed=87004,
            maximum_programming_pulses=2,
            verify_tolerance_x=0.01,
        ),
        data=SimpleNamespace(num_points=4),
        evaluation=SimpleNamespace(
            maximum_validation_batches=1,
            sample_limit=None,
        ),
    )


def _context(tmp_path, inputs, labels, *, tiki_taka: bool):
    predeployment, _ = runtime._ordered_labeled_cohort_report(
        inputs=[inputs],
        labels=[labels],
        split="predeployment_offchip_first_epoch_update_stream",
    )
    context = {
        "faulted_checkpoint_path": tmp_path / "faulted.pt",
        "recovery_state_path": tmp_path / "recovery.pt",
        "fault_mask": torch.tensor(
            [True, False, False, False, False, False, False, False]
        ),
        "stuck_persistent_q": torch.tensor(
            [0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ),
        "fault_report": {"policy": "test_published_companion"},
        "fault_source_population_fingerprint": "test-published-slow",
        "predeployment_training_cohort": predeployment,
    }
    if tiki_taka:
        context.update(
            {
                "fast_population": _population(
                    assignment_seed=88004,
                    corruption_policy="published",
                ),
                "fast_population_receipt": {
                    "bound_treatment": {
                        "source_population_fingerprint": "test-fast-source"
                    }
                },
                "fast_endpoint_seed": 98402,
                "fast_commissioned_checkpoint_path": (
                    tmp_path / "fast_commissioned.pt"
                ),
            }
        )
    return context


def _fake_evaluation(*, plant, **_kwargs):
    value = float(plant.apparent.mean().item())
    row = {"student_accuracy": value, "examples": 4}
    return {
        "network_forward_state": "apparent_q",
        "hidden_update_state": "persistent_q",
        "apparent_forward": dict(row),
        "persistent_diagnostic": dict(row),
    }


def test_direct_runtime_is_chronological_local_reloadable_and_pulse_exact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(runtime, "_evaluate_plant_states", _fake_evaluation)

    def forbidden_autograd(*_args, **_kwargs):
        raise AssertionError("autograd is forbidden in manual on-chip BP")

    monkeypatch.setattr(torch.autograd, "grad", forbidden_autograd)
    plant = _plant()
    p0 = plant.state_dict()
    inputs, labels = _batch()
    context = _context(tmp_path, inputs, labels, tiki_taka=False)

    report, final_state = runtime._recover(
        plant=plant,
        spec=_spec("supervised_ce_stochastic_pulse_sgd"),
        endpoint_seed=89402,
        layout=_layout(),
        digital_scales=(0.5, 0.4),
        teacher=SimpleNamespace(),
        validation_loader=[(inputs, labels)],
        train_loader=[(inputs, labels)],
        device=torch.device("cpu"),
        post_fault_context=context,
    )

    signature = inspect.signature(
        runtime._run_supervised_stochastic_on_chip_recovery
    )
    for forbidden in ("teacher", "fault_mask", "autograd", "shadow"):
        assert forbidden not in signature.parameters
    faulted = torch.load(tmp_path / "faulted.pt", weights_only=True)
    assert faulted["fault_transition"]["pre_fault_persistent_sha256"] == (
        runtime.tensor_sha256(p0["persistent"])
    )
    mask = context["fault_mask"]
    assert torch.equal(final_state["persistent"][mask], faulted["persistent"][mask])
    assert report["optimizer"]["fast_commanded_pulses"] == 0
    assert report["optimizer"]["slow_commanded_pulses"] > 0
    assert report["optimizer"]["slow_commanded_pulses"] == (
        report["posthoc_pulse_effects"]["commanded_pulses"]
    )
    assert report["optimizer"]["repair_cohort"]["examples"] == 4
    assert report["optimizer"]["state_dict_reload_bit_exact"] is True
    assert report["recovery_state"]["artifact_reload_bit_exact"] is True
    saved = torch.load(tmp_path / "recovery.pt", weights_only=True)
    assert saved["updater_state"]["contract"]["teacher_queries"] == 0
    assert saved["updater_state"]["contract"]["autograd_calls"] == 0
    assert saved["updater_state"]["contract"]["fault_mask_exposed"] is False
    assert "fault_mask" not in saved["updater_state"]
    assert report["tiki_taka_fast_array"] is None


def test_tiki_taka_runtime_physically_commissions_fast_array_then_transfers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(runtime, "_evaluate_plant_states", _fake_evaluation)
    plant = _plant()
    inputs, labels = _batch()
    context = _context(tmp_path, inputs, labels, tiki_taka=True)

    report, final_state = runtime._recover(
        plant=plant,
        spec=_spec("supervised_ce_tiki_taka_v1"),
        endpoint_seed=89402,
        layout=_layout(),
        digital_scales=(0.5, 0.4),
        teacher=SimpleNamespace(),
        validation_loader=[(inputs, labels)],
        train_loader=[(inputs, labels)],
        device=torch.device("cpu"),
        post_fault_context=context,
    )

    assert (tmp_path / "faulted.pt").is_file()
    assert (tmp_path / "fast_commissioned.pt").is_file()
    assert (tmp_path / "recovery.pt").is_file()
    fast = report["tiki_taka_fast_array"]
    assert fast["visible_to_network_forward"] is False
    target = fast["initialization"]["target"]
    assert target["policy"] == "strict_q_zero_target_mask_blind"
    assert target["target_all_zero"] is True
    assert target["target_q_sha256"] == runtime.tensor_sha256(
        torch.zeros(context["fast_population"].size, dtype=torch.float32)
    )
    assert target["target_construction"] == {
        "requested_q": "strict_zero_for_every_physical_cell",
        "uses_per_cell_support": False,
        "uses_corrupt_mask": False,
        "uses_sampled_stuck_values": False,
        "mutates_array": False,
    }
    assert target["posthoc_support_diagnostic"][
        "analysis_only_not_used_by_target_or_controller"
    ] is True
    assert target["posthoc_support_diagnostic"][
        "corrupt_cells_with_target_outside_support"
    ] == 1
    assert fast["initialization"]["requested_target"] == (
        "literal_q_zero_for_every_fast_cell"
    )
    assert fast["initialization"]["controller_accesses_fault_mask"] is False
    assert fast["initialization"]["eligible_mask_supplied"] is False
    assert fast["initialization"]["program_verify"]["budget_exhausted"] >= 1
    assert fast["initialization"]["receipt"]["physical_program_verify"] is True
    assert fast["initialization"]["receipt"]["policy"] == (
        "program_verify_strict_q_zero_mask_blind"
    )
    assert fast["initialization"]["receipt"]["direct_state_assignment"] is False
    assert fast["initialization"][
        "commissioning_pulses_excluded_from_recovery_pulse_counts"
    ] is True
    assert report["optimizer"]["transfer_events"] == 1
    assert report["optimizer"]["fast_commanded_pulses"] > 0
    assert report["optimizer"]["slow_commanded_pulses"] > 0
    assert report["optimizer"]["slow_commanded_pulses"] == (
        report["posthoc_pulse_effects"]["commanded_pulses"]
    )
    assert report["optimizer"]["fast_commanded_pulses"] == fast[
        "posthoc_recovery_pulse_effects"
    ]["commanded_pulses"]
    assert report["on_chip_backprop_contract"]["final_program_verify"] is False
    faulted = torch.load(tmp_path / "faulted.pt", weights_only=True)
    mask = context["fault_mask"]
    assert torch.equal(final_state["persistent"][mask], faulted["persistent"][mask])

    commissioned = torch.load(tmp_path / "fast_commissioned.pt", weights_only=True)
    recovery = torch.load(tmp_path / "recovery.pt", weights_only=True)
    fast_mask = context["fast_population"].corrupt
    fast_initial_q = commissioned["plant_state"]["persistent"]
    fast_final_q = recovery["fast_array"]["final_plant_state"]["persistent"]
    assert fast_initial_q[fast_mask].tolist() == pytest.approx([0.2])
    assert torch.equal(fast_initial_q[fast_mask], fast_final_q[fast_mask])
    assert recovery["updater_state"]["tile_cursors"].tolist() == [1, 1]
    assert report["recovery_state"]["artifact_reload_bit_exact"] is True
