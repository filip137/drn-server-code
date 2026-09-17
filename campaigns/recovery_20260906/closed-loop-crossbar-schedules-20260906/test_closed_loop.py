"""Numerical tests for the feedback controller, including hidden stuck cells."""
import torch
from closed_loop import ClosedLoopAdam


def test_subtolerance_updates_accumulate_and_zero_updates_hold_state():
    apparent = torch.tensor([0.0, 0.4], device='cuda')
    writes = []
    def pulse(direction):
        writes.append(direction.clone())
        apparent.add_(direction * 0.05)
    optimizer = ClosedLoopAdam(lambda: apparent, pulse, learning_rate=.01, tolerance=.025)
    original = apparent.clone()
    optimizer.step(torch.zeros_like(apparent))
    assert torch.equal(original, apparent) and not writes
    for _ in range(5):
        optimizer.step(torch.tensor([-1., 0.], device='cuda'))
    assert writes and apparent[1] == original[1]
    assert abs(optimizer.target[0] - apparent[0]) <= .025


def test_feedback_reverses_overshoot_and_stops_at_apparent_acceptance():
    apparent = torch.tensor([0.], device='cuda')
    sequence = iter([.30, .11])
    directions = []
    def pulse(direction):
        directions.append(int(direction[0]))
        apparent.fill_(next(sequence))
    optimizer = ClosedLoopAdam(lambda: apparent, pulse, learning_rate=.1, tolerance=.02)
    result = optimizer.step(torch.tensor([-1.], device='cuda'))
    assert directions == [1, -1] and result['pulses'] == 2


def test_unresponsive_plant_exhausts_own_budget_without_fault_information():
    apparent = torch.tensor([0.], device='cuda')
    calls = []
    def pulse(direction):
        calls.append(direction.clone())  # Unresponsive physical plant.
    optimizer = ClosedLoopAdam(lambda: apparent, pulse, learning_rate=.1, tolerance=.02, maximum_pulses=2, total_pulse_cap=3)
    first = optimizer.step(torch.tensor([-1.], device='cuda'))
    second = optimizer.step(torch.tensor([-1.], device='cuda'))
    third = optimizer.step(torch.tensor([-1.], device='cuda'))
    assert first['pulses'] == 2 and optimizer.burst_exhaustions == 1
    assert second['pulses'] == 1 and third['pulses'] == 0
    assert len(calls) == 3 and apparent[0] == 0
    assert optimizer.pulse_count[0] == 3


if __name__ == '__main__':
    assert torch.cuda.is_available()
    for name, function in list(globals().items()):
        if name.startswith('test_'):
            function()
            print(name, 'PASS', flush=True)
