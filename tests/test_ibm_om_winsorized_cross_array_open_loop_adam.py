from __future__ import annotations

import torch

from experiments.mnist_relu_drn.ibm_om_baseline_selection import quad_stack
from experiments.mnist_relu_drn.ibm_om_winsorized_cross_array_open_loop_adam import (
    ColumnSerialOpenLoopAdam,
    build_target_baseline_contrast_transfer,
    open_loop_pulse_port,
)
from training.ibm_reram_program_verify import (
    IbmReramPopulation,
    IbmReramRawActivePlant,
)


def _source_payload() -> dict:
    full = torch.tensor(((0.8, 0.4), (0.4, 0.8)), dtype=torch.float32)
    return {
        "binding_shapes": [[2, 2], [2, 2]],
        "plant_continuation_state": {
            "persistent": torch.cat((full.reshape(-1), full.reshape(-1))) - 1.0,
        },
    }


def test_unclipped_transfer_preserves_contrast_and_marks_unsupported_weight() -> None:
    baseline = torch.full((2, 2), 0.2, dtype=torch.float32)
    lower = torch.zeros((2, 2), dtype=torch.float32)
    upper_supported = torch.full((2, 2), 0.5, dtype=torch.float32)
    upper_unsupported = torch.full((2, 2), 0.3, dtype=torch.float32)
    transfer = build_target_baseline_contrast_transfer(
        _source_payload(),
        target_baseline_raw_x=(baseline, baseline),
        target_lower_raw_x=(lower, lower),
        target_upper_raw_x=(upper_supported, upper_unsupported),
    )
    assert transfer.report["unsupported_logical_weights"] == 1
    assert transfer.target_raw_x[1].max().item() > upper_unsupported.max().item()
    for target, layout in zip(transfer.target_full_conductance, ("halves", "paired")):
        quad = quad_stack(target, layout=layout)
        contrast = (quad[..., 0] - quad[..., 1] - quad[..., 2] + quad[..., 3]) / 2
        assert torch.allclose(contrast, torch.tensor(((0.4,),)))


class _RecordingPort:
    size = 8

    def __init__(self) -> None:
        self.requests = []

    def apply_identical_pulses(self, directions, counts) -> None:
        self.requests.append((directions.clone(), counts.clone()))


def test_column_serial_emulator_has_exact_bernoulli_and_phase_accounting() -> None:
    port = _RecordingPort()
    optimizer = ColumnSerialOpenLoopAdam(
        port,
        device="cpu",
        binding_shapes=((2, 2), (2, 2)),
        learning_rate_raw_x=1.0,
        nominal_delta_x=0.1,
        pulse_cap=1,
        pulse_selection_seed=3,
    )
    gradient = torch.ones((2, 2), dtype=torch.float32)
    step = optimizer.step((gradient, gradient))
    assert step.conceptual_column_phases == 4
    assert step.requested_cell_coincidences == 8
    assert step.applied_cell_pulses == 8
    assert len(port.requests) == 1
    directions, counts = port.requests[0]
    assert torch.equal(directions, -torch.ones_like(directions))
    assert torch.equal(counts, torch.ones_like(counts))
    state = optimizer.state_dict()
    assert state["verify_reads_during_updates"] == 0
    assert state["fully_parallel_ibm_outer_product_equivalent"] is False
    assert state["total_conceptual_column_phases"] == 4


def test_open_loop_port_exposes_no_verify_read() -> None:
    size = 8
    vector = lambda value: torch.full((size,), value, dtype=torch.float32)
    population = IbmReramPopulation(
        preset="reram_array_om",
        aihwkit_version="1.1.0",
        nominal_dw_min=0.1,
        dw_min_std=0.0,
        write_noise_std=0.0,
        mult_noise=False,
        construction_seeds=torch.arange(1, size + 1, dtype=torch.int64),
        max_bound=vector(1.0),
        min_bound=vector(-1.0),
        dwmin_up=vector(0.1),
        dwmin_down=vector(0.1),
        reference=vector(0.0),
        corrupt=torch.zeros(size, dtype=torch.bool),
        preset_parameters={},
    )
    plant = IbmReramRawActivePlant(
        population, seeds=tuple(range(11, 19)), device="cpu"
    )
    plant.initialize_at_sampled_lower()
    port = open_loop_pulse_port(plant)
    assert not hasattr(port, "verify")
    before = plant.persistent.clone()
    port.apply_identical_pulses(
        torch.ones(size, dtype=torch.int8), torch.ones(size, dtype=torch.int64)
    )
    assert bool(torch.all(plant.persistent > before))
