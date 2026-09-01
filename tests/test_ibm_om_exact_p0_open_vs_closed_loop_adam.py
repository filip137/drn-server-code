from __future__ import annotations

import pytest
import torch

from experiments.mnist_relu_drn.ibm_om_exact_p0_open_vs_closed_loop_adam import (
    IncrementalOnePulseProgramVerifyAdam,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam import (
    ColumnSerialOpenLoopAdam,
)
from experiments.mnist_relu_drn.ibm_om_winsorized_onchip_adam import flatten_physical


class FakeVerifyPort:
    def __init__(self, raw_x: torch.Tensor, *, immobile: torch.Tensor | None = None) -> None:
        self.raw_x = raw_x.clone()
        self.immobile = (
            torch.zeros_like(raw_x, dtype=torch.bool)
            if immobile is None
            else immobile.clone()
        )
        self.verify_calls = 0
        self.pulse_rounds = 0

    @property
    def size(self) -> int:
        return int(self.raw_x.numel())

    def verify(self) -> torch.Tensor:
        self.verify_calls += 1
        return self.raw_x.clone()

    def apply_identical_pulses(
        self, directions: torch.Tensor, counts: torch.Tensor
    ) -> None:
        assert int(counts.max().item()) <= 1
        selected = (counts != 0) & ~self.immobile
        self.raw_x[selected] += 0.1 * directions[selected].to(torch.float32)
        self.pulse_rounds += 1


class FakeOpenLoopPort:
    def __init__(self, size: int) -> None:
        self._size = size

    @property
    def size(self) -> int:
        return self._size

    def apply_identical_pulses(
        self, directions: torch.Tensor, counts: torch.Tensor
    ) -> None:
        del directions, counts


def test_cached_incremental_pv_projects_targets_and_reads_only_after_pulse() -> None:
    initial_x = torch.tensor([-0.1, 0.5, 1.2], dtype=torch.float32)
    port = FakeVerifyPort(initial_x, immobile=torch.tensor([False, False, True]))
    optimizer = IncrementalOnePulseProgramVerifyAdam(
        port,
        initial_apparent_raw_a=2.0 * initial_x - 1.0,
        device="cpu",
        binding_shapes=((2, 1), (1, 1)),
        learning_rate_raw_x=0.2,
        nominal_delta_x=0.1,
        verify_tolerance_raw_x=0.05,
        pulse_cap=2,
    )
    assert port.verify_calls == 0
    assert optimizer.initial_target_projection.cells == 2
    assert [value.cells for value in optimizer.initial_target_projection_by_layer] == [1, 1]

    step = optimizer.step((torch.full((2, 1), -1.0), torch.full((1, 1), -1.0)))
    assert step.issued_cell_pulses == 3
    assert step.api_full_port_verify_calls == 1
    assert step.conceptual_post_pulse_cell_observations == 3
    assert port.verify_calls == 1
    assert int(optimizer.pulse_count.sum().item()) == 3
    assert optimizer.cached_apparent_raw_x[2].item() == pytest.approx(1.2)
    assert step.target_projection.cells >= 1
    assert optimizer.state_dict()["persistent_state_visible_to_controller"] is False


def test_closed_loop_uses_the_exact_open_loop_digital_adam_command() -> None:
    shapes = ((2, 1), (1, 1))
    gradients = (torch.tensor([[0.2], [-0.4]]), torch.tensor([[0.6]]))
    open_optimizer = ColumnSerialOpenLoopAdam(
        FakeOpenLoopPort(3),
        device="cpu",
        binding_shapes=shapes,
        learning_rate_raw_x=1e-4,
        nominal_delta_x=0.04745,
        pulse_cap=64,
        pulse_selection_seed=94401,
    )
    expected = open_optimizer._adam_command(
        flatten_physical(tuple(2.0 * value for value in gradients))
    )
    port = FakeVerifyPort(torch.full((3,), 0.5))
    closed_optimizer = IncrementalOnePulseProgramVerifyAdam(
        port,
        initial_apparent_raw_a=torch.zeros(3),
        device="cpu",
        binding_shapes=shapes,
        learning_rate_raw_x=1e-4,
        nominal_delta_x=0.04745,
        verify_tolerance_raw_x=1.0,
        pulse_cap=64,
    )
    closed_optimizer.step(gradients)
    torch.testing.assert_close(
        closed_optimizer.desired_raw_x - 0.5,
        expected,
        rtol=1e-5,
        atol=3e-8,
    )
    assert torch.equal(closed_optimizer.first_moment, open_optimizer.first_moment)
    assert torch.equal(closed_optimizer.second_moment, open_optimizer.second_moment)
    assert port.verify_calls == 0


def test_closed_loop_checkpoint_roundtrip_restores_controller_cache_and_rejects_adam_drift() -> None:
    shapes = ((2, 1), (1, 1))
    port = FakeVerifyPort(torch.full((3,), 0.5))
    optimizer = IncrementalOnePulseProgramVerifyAdam(
        port,
        initial_apparent_raw_a=torch.zeros(3),
        device="cpu",
        binding_shapes=shapes,
        learning_rate_raw_x=0.1,
        nominal_delta_x=0.04745,
        verify_tolerance_raw_x=0.01,
        pulse_cap=4,
    )
    optimizer.step((torch.ones((2, 1)), torch.ones((1, 1))))
    state = optimizer.state_dict()
    replay = IncrementalOnePulseProgramVerifyAdam(
        FakeVerifyPort(torch.full((3,), 0.5)),
        initial_apparent_raw_a=torch.zeros(3),
        device="cpu",
        binding_shapes=shapes,
        learning_rate_raw_x=0.1,
        nominal_delta_x=0.04745,
        verify_tolerance_raw_x=0.01,
        pulse_cap=4,
    )
    replay.load_state_dict(state)
    assert torch.equal(replay.cached_apparent_raw_x, optimizer.cached_apparent_raw_x)
    assert torch.equal(replay.desired_raw_x, optimizer.desired_raw_x)
    assert torch.equal(replay.pulse_count, optimizer.pulse_count)

    drifted = dict(state)
    drifted["beta1"] = 0.8
    with pytest.raises(ValueError, match="checkpoint contract"):
        replay.load_state_dict(drifted)


def test_partial_mask_freezes_adam_state_target_and_pulse_eligibility() -> None:
    shapes = ((2, 1), (1, 1))
    initial_x = torch.tensor([-0.1, 0.5, 1.2], dtype=torch.float32)
    writable = torch.tensor([False, True, False])
    port = FakeVerifyPort(initial_x)
    optimizer = IncrementalOnePulseProgramVerifyAdam(
        port,
        initial_apparent_raw_a=2.0 * initial_x - 1.0,
        device="cpu",
        binding_shapes=shapes,
        learning_rate_raw_x=0.2,
        nominal_delta_x=0.1,
        verify_tolerance_raw_x=0.01,
        pulse_cap=4,
        trainable_cell_mask=writable,
    )
    frozen_target = optimizer.desired_raw_x[~writable].clone()
    step = optimizer.step(
        (torch.full((2, 1), -1.0), torch.full((1, 1), -1.0))
    )
    assert step.issued_cell_pulses == 1
    assert step.frozen_target_debt_cells == 2
    assert torch.equal(optimizer.pulse_count[~writable], torch.zeros(2, dtype=torch.int64))
    assert torch.equal(optimizer.first_moment[~writable], torch.zeros(2))
    assert torch.equal(optimizer.second_moment[~writable], torch.zeros(2))
    assert torch.equal(optimizer.desired_raw_x[~writable], frozen_target)
    assert port.raw_x[0].item() == pytest.approx(-0.1)
    assert port.raw_x[2].item() == pytest.approx(1.2)

    wrong_mask = IncrementalOnePulseProgramVerifyAdam(
        FakeVerifyPort(initial_x),
        initial_apparent_raw_a=2.0 * initial_x - 1.0,
        device="cpu",
        binding_shapes=shapes,
        learning_rate_raw_x=0.2,
        nominal_delta_x=0.1,
        verify_tolerance_raw_x=0.01,
        pulse_cap=4,
        trainable_cell_mask=torch.tensor([True, False, False]),
    )
    with pytest.raises(ValueError, match="checkpoint contract"):
        wrong_mask.load_state_dict(optimizer.state_dict())
