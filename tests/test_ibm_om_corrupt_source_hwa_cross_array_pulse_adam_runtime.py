from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_corrupt_source_hwa_cross_array_pulse_adam_runtime import (
    _assert_corrupt_immobile,
    _assert_distinct_array_pair,
    _assert_p0_roundtrip,
    _deployment_protocol_adapter,
    _development_key,
    _hwa_checkpoint_payload,
    _mean2_objective,
    _paired_target_hwa_gate,
    _positive_full_g,
    _require_protocol_contract,
    _recovery_gate,
    _select_trained_hwa_epoch,
)
from experiments.mnist_relu_drn.ibm_om_corrupt_source_hwa_cross_array_pulse_adam_config import (
    _protocol,
)
from training.checkpoint import atomic_torch_save


def test_mean2_objective_is_exact_quarter_plus_two_three_eighths() -> None:
    assert _mean2_objective(2.0, (4.0, 8.0)) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="exactly two"):
        _mean2_objective(2.0, (4.0,))


def test_development_selection_uses_only_declared_endpoint_statistics() -> None:
    base = {
        "persistent_mean_accuracy": 0.80,
        "persistent_minimum_accuracy": 0.70,
        "persistent_mean_kl_teacher_student": 0.20,
    }
    assert _development_key({**base, "persistent_mean_accuracy": 0.81}, 3) > (
        _development_key({**base, "persistent_minimum_accuracy": 0.99}, 1)
    )
    assert _development_key({**base, "persistent_minimum_accuracy": 0.71}, 3) > (
        _development_key(base, 1)
    )
    assert _development_key(
        {**base, "persistent_mean_kl_teacher_student": 0.19}, 3
    ) > _development_key(base, 1)
    assert _development_key(base, 1) > _development_key(base, 2)

    # Regression: if epoch 1 is the selected point and epoch 3 regresses, the
    # final epoch must not replace it merely because training ran longer.
    epoch_1 = {**base, "persistent_mean_accuracy": 0.82}
    epoch_3 = {**base, "persistent_mean_accuracy": 0.79}
    assert _development_key(epoch_1, 1) > _development_key(epoch_3, 3)


def test_selected_hwa_checkpoint_is_master_only_not_mismatched_optimizer_resume() -> None:
    selected = (
        torch.tensor([[0.25]], dtype=torch.float32),
        torch.tensor([[-0.50]], dtype=torch.float32),
    )
    payload = _hwa_checkpoint_payload(
        epoch=1,
        masters=selected,
        source_absmax=(2.0, 3.0),
        development={
            "persistent_mean_accuracy": 0.82,
            "persistent_minimum_accuracy": 0.80,
            "persistent_mean_kl_teacher_student": 0.1,
        },
        teacher_sha256="a" * 64,
        source_assignment_seed=93001,
        source_published_fingerprint="published",
        source_repaired_fingerprint="repaired",
        global_minibatch_ordinal=3750,
    )
    assert payload["selected_epoch"] == 1
    assert payload["global_minibatch_ordinal"] == 3750
    assert payload["checkpoint_role"] == (
        "selected_logical_master_only_not_optimizer_resume"
    )
    assert payload["optimizer_continuation_state"] is None
    assert "optimizer_state" not in payload
    torch.testing.assert_close(payload["logical_master"][0], selected[0])


def test_epoch0_control_cannot_win_hwa_checkpoint_selection() -> None:
    epoch0 = {
        "persistent_mean_accuracy": 0.90,
        "persistent_minimum_accuracy": 0.88,
        "persistent_mean_kl_teacher_student": 0.1,
    }
    trained = (
        (1, {**epoch0, "persistent_mean_accuracy": 0.70}),
        (2, {**epoch0, "persistent_mean_accuracy": 0.80}),
        (3, {**epoch0, "persistent_mean_accuracy": 0.75}),
    )
    selected = _select_trained_hwa_epoch(trained)
    assert selected == 2
    assert epoch0["persistent_mean_accuracy"] > trained[1][1]["persistent_mean_accuracy"]
    with pytest.raises(ValueError, match="trained epochs"):
        _select_trained_hwa_epoch(((0, epoch0), *trained))


def test_circuit_gate_rejects_negative_or_out_of_range_conductance() -> None:
    _positive_full_g(
        (
            torch.tensor([[0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[1.5, 2.0]], dtype=torch.float32),
        )
    )
    with pytest.raises(ValueError, match=r"G=\[0,2\]"):
        _positive_full_g(
            (
                torch.tensor([[-0.01]], dtype=torch.float32),
                torch.tensor([[1.0]], dtype=torch.float32),
            )
        )
    with pytest.raises(ValueError, match=r"G=\[0,2\]"):
        _positive_full_g(
            (
                torch.tensor([[0.0]], dtype=torch.float32),
                torch.tensor([[2.01]], dtype=torch.float32),
            )
        )


def test_corrupt_immobility_gate_checks_only_published_stuck_cells() -> None:
    before = torch.tensor((-0.2, 0.1, 0.3), dtype=torch.float32)
    after = torch.tensor((-0.2, 0.8, 0.3), dtype=torch.float32)
    mask = torch.tensor((True, False, True))
    report = _assert_corrupt_immobile(
        before, after, mask, context="unit_test"
    )
    assert report["corrupt_cells"] == 2
    assert report["changed_corrupt_cells"] == 0
    assert report["stuck_cells_immobile"] is True

    changed = after.clone()
    changed[2] += 0.01
    with pytest.raises(RuntimeError, match="corrupt cells moved"):
        _assert_corrupt_immobile(before, changed, mask, context="unit_test")


def test_target_assignment_must_be_a_distinct_physical_draw() -> None:
    def pair(seed: int, published: str, repaired: str, construction: tuple[int, int]):
        population = SimpleNamespace(
            binding_keys=("w1", "w2"),
            binding_shapes=((2, 2), (2, 2)),
            binding_sampling_seeds=construction,
            donor_sampling_seeds=(construction[0] + 100, construction[1] + 100),
            fingerprint=published,
        )
        return SimpleNamespace(
            assignment_seed=seed,
            published=population,
            repaired=SimpleNamespace(fingerprint=repaired),
        )

    source = pair(93001, "source-p", "source-r", (11, 12))
    target = pair(93002, "target-p", "target-r", (21, 22))
    report = _assert_distinct_array_pair(source, target)
    assert report["assignment_seeds_differ"] is True
    assert report["binding_construction_seeds_differ_coordinatewise"] is True

    reused = pair(93002, "target-p", "target-r", (11, 22))
    with pytest.raises(RuntimeError, match="not a distinct physical"):
        _assert_distinct_array_pair(source, reused)


def test_p0_roundtrip_requires_exact_persistent_and_rng_state(tmp_path) -> None:
    state = {
        "schema": "plant",
        "schema_version": 2,
        "state_coordinate": "native_raw_active_a",
        "preset": "reram_array_om",
        "rng_backend": "per_trajectory_buffered_torch_cpu",
        "maximum_random_draws": 385,
        "construction_seeds": torch.tensor((1, 2), dtype=torch.int64),
        "persistent": torch.tensor((-0.2, 0.3), dtype=torch.float32),
        "apparent": torch.tensor((-0.1, 0.4), dtype=torch.float32),
        "draw_indices": torch.tensor((3, 5), dtype=torch.int64),
        "seeds": (11, 12),
    }
    payload = {
        "schema": "deployment",
        "schema_version": 1,
        "assignment_seed": 93002,
        "endpoint_seed": 93301,
        "target_kind": "selected_hwa",
        "maximum_program_pulses": 128,
        "maximum_random_draws": 385,
        "target_raw_x": torch.tensor((0.4, 0.6), dtype=torch.float32),
        "continuation_state": state,
    }
    path = atomic_torch_save(payload, tmp_path / "p0.pt")
    observed = torch.load(path, map_location="cpu", weights_only=True)
    report = _assert_p0_roundtrip(payload, observed)
    assert report["recovery_consumes_reloaded_artifact"] is True

    observed["continuation_state"]["draw_indices"][0] += 1
    with pytest.raises(RuntimeError, match="draw_indices"):
        _assert_p0_roundtrip(payload, observed)


def test_paired_target_hwa_and_recovery_gates_are_explicit() -> None:
    epoch0 = []
    selected = []
    for index, seed in enumerate((93301, 93302, 93303, 93304, 93305)):
        before = 0.70 + 0.001 * index
        gain = 0.03 if index < 4 else -0.01
        epoch0.append(
            {"endpoint_seed": seed, "test": {"student_accuracy": before}}
        )
        selected.append(
            {
                "endpoint_seed": seed,
                "test": {"student_accuracy": before + gain},
            }
        )
    hwa_gate = _paired_target_hwa_gate(epoch0, selected)
    assert hwa_gate["wins"] == 4
    assert hwa_gate["mean_selected_minus_epoch0_accuracy"] == pytest.approx(
        0.022
    )
    assert hwa_gate["passed"] is True

    recovery = {
        "before_test": {"student_accuracy": 0.75},
        "after_test": {"student_accuracy": 0.91},
        "corrupt_immobility": {"stuck_cells_immobile": True},
        "pulse": {"verify_reads_during_updates": 0},
    }
    recovery_result = _recovery_gate(
        recovery, selected_source_persistent_mean_accuracy=0.92
    )
    assert recovery_result["post_minus_pre_accuracy"] == pytest.approx(0.16)
    assert recovery_result["source_recovery_floor_passed"] is True
    assert recovery_result["within_symmetric_source_gap_diagnostic"] is True
    assert recovery_result["physical_safety_passed"] is True
    assert recovery_result["passed"] is True


@pytest.mark.parametrize(
    (
        "source_accuracy",
        "post_accuracy",
        "floor_passed",
        "symmetric_diagnostic",
    ),
    (
        # Beneficial 6.3275 pp overshoot passes the one-sided floor even
        # though it is not symmetric parity within 2 pp.
        (0.839225, 0.9025, True, False),
        # A 1.9 pp shortfall remains inside the allowed recovery floor.
        (0.90, 0.881, True, True),
        # A 2.1 pp shortfall fails the recovery floor.
        (0.90, 0.879, False, False),
    ),
)
def test_recovery_source_floor_is_one_sided(
    source_accuracy: float,
    post_accuracy: float,
    floor_passed: bool,
    symmetric_diagnostic: bool,
) -> None:
    recovery = {
        "before_test": {"student_accuracy": 0.70},
        "after_test": {"student_accuracy": post_accuracy},
        "corrupt_immobility": {"stuck_cells_immobile": True},
        "pulse": {"verify_reads_during_updates": 0},
    }
    report = _recovery_gate(
        recovery,
        selected_source_persistent_mean_accuracy=source_accuracy,
    )
    assert report["source_recovery_floor_passed"] is floor_passed
    assert (
        report["within_symmetric_source_gap_diagnostic"]
        is symmetric_diagnostic
    )
    assert report["passed"] is floor_passed


def test_protocol_adapter_preserves_p0_and_zero_verify_recovery_contract() -> None:
    protocol = SimpleNamespace(
        target=SimpleNamespace(
            assignment_seed=93002,
            endpoint_seeds=(93301, 93302, 93303, 93304, 93305),
            fine_tune_endpoint_seed=93301,
        ),
        deployment=SimpleNamespace(
            maximum_random_draws=385,
            maximum_program_pulses=128,
            verify_tolerance_raw_x=0.023725,
        ),
        recovery=SimpleNamespace(
            learning_rate_raw_x=3e-5,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            epochs=1,
            maximum_batches=None,
            pulse_cap=64,
            pulse_selection_seed=93401,
            conceptual_column_phases_per_minibatch=(100, 20),
            vectorization_equivalence=(
                "disjoint_cell_state_and_per_cell_RNG_equivalent"
            ),
        ),
    )
    adapted = _deployment_protocol_adapter(protocol)
    assert adapted.target.assignment_seed == 93002
    assert adapted.target.fine_tune_endpoint_seed == 93301
    assert adapted.recovery.learning_rate_raw_x == pytest.approx(3e-5)
    assert adapted.recovery.pulse_cap == 64
    assert adapted.recovery.epochs == 1


def test_strict_config_resolves_the_frozen_source_target_ladder() -> None:
    contract = _require_protocol_contract(_protocol())
    assert contract["source_assignment_seed"] == 93001
    assert contract["source_training_endpoint_seeds"] == (
        93201,
        93202,
        93203,
        93204,
    )
    assert contract["source_selection_endpoint_seeds"] == (
        93221,
        93222,
        93223,
        93224,
    )
    assert contract["target_assignment_seed"] == 93002
    assert contract["target_endpoint_seeds"] == (
        93301,
        93302,
        93303,
        93304,
        93305,
    )
    assert contract["p0_endpoint_seed"] == 93301
    assert contract["hwa_epochs"] == 3
    assert contract["level_spacing_raw_x"] == pytest.approx(0.04745)
