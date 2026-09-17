import pytest
import torch

from closed_loop import ClosedLoopAdam
from run_recovery import schedule_factor


def test_epoch_ten_floor_and_later_epochs():
    schedule = dict(kind='exponential', final_factor=.01, decay_epochs=10)
    assert schedule_factor(schedule, 1) == 1
    assert schedule_factor(schedule, 10) == .01
    assert schedule_factor(schedule, 30) == .01
    assert schedule_factor(schedule, 2) == pytest.approx(.599484250318941)
    with pytest.raises(ValueError):
        schedule_factor(schedule, 0)


def test_decay_scales_accumulated_target_without_resetting_moments():
    assert torch.cuda.is_available(), 'Expected the actual CUDA test gate.'
    apparent = torch.zeros(4, device='cuda')
    writes = []

    def pulse(direction):
        writes.append(direction.clone())

    optimizer = ClosedLoopAdam(lambda: apparent, pulse, learning_rate=.01,
                               tolerance=1., maximum_pulses=128, total_pulse_cap=640)
    gradient = torch.ones_like(apparent)
    optimizer.step(gradient)
    first_target = optimizer.target.clone()
    optimizer.learning_rate_scale = .01
    optimizer.step(gradient)
    assert optimizer.step_index == 2
    assert torch.allclose(optimizer.target, first_target * 1.01)
    assert torch.allclose(optimizer.first_moment, torch.full_like(apparent, .19))
    assert not writes
    assert optimizer.state_dict()['learning_rate_scale'] == .01
