from __future__ import annotations

import torch

from experiments.mnist_relu_drn.ibm_om_winsorized_endpoint_optimizer_recovery import (
    ColumnSerialOpenLoopSgd,
    OmPlantTikiTaka,
    canonical_input_group_indices,
    initialize_auxiliary_at_symmetry,
    intrinsic_symmetry_raw_a,
)
from training.ibm_reram_program_verify import (
    IbmReramPopulation,
    IbmReramRawActivePlant,
)


def _population(size: int, *, up: float = 0.1, down: float = 0.1) -> IbmReramPopulation:
    vector = lambda value: torch.full((size,), value, dtype=torch.float32)
    return IbmReramPopulation(
        preset="reram_array_om",
        aihwkit_version="1.1.0",
        nominal_dw_min=0.1,
        dw_min_std=0.0,
        write_noise_std=0.0,
        mult_noise=False,
        construction_seeds=torch.arange(1, size + 1, dtype=torch.int64),
        max_bound=vector(1.0),
        min_bound=vector(-1.0),
        dwmin_up=vector(up),
        dwmin_down=vector(down),
        reference=vector(0.0),
        corrupt=torch.zeros(size, dtype=torch.bool),
        preset_parameters={},
    )


class _RecordingPort:
    def __init__(self, size: int) -> None:
        self.size = size
        self.requests: list[tuple[torch.Tensor, torch.Tensor]] = []

    def apply_identical_pulses(
        self, directions: torch.Tensor, counts: torch.Tensor
    ) -> None:
        self.requests.append((directions.clone(), counts.clone()))


def test_intrinsic_symmetry_solves_asymmetric_pulse_response_without_clamp() -> None:
    population = _population(3, up=0.2, down=0.1)
    result = intrinsic_symmetry_raw_a(population)
    # 0.2*(1-a) == 0.1*(1+a) gives a=1/3.
    assert torch.allclose(result.raw_a, torch.full((3,), 1.0 / 3.0))
    assert result.report["numerically_clamped_cells"] == 0
    assert result.report["materially_out_of_support_cells"] == 0
    assert result.report["maximum_pulse_response_mismatch"] < 1e-12


def test_canonical_transfer_group_is_dual_source_rail_rows_not_destination_column() -> None:
    first = canonical_input_group_indices((8, 6), 1, device="cpu")
    # Logical input 1 maps to stored physical rows 1 and 1+4, all 6 rails.
    assert torch.equal(
        first,
        torch.tensor(
            [6, 7, 8, 9, 10, 11, 30, 31, 32, 33, 34, 35],
            dtype=torch.int64,
        ),
    )
    assert canonical_input_group_indices((1568, 100), 783, device="cpu").numel() == 200
    assert canonical_input_group_indices((100, 20), 49, device="cpu").numel() == 40


def test_open_loop_sgd_uses_raw_x_chain_rule_and_has_no_verify_surface() -> None:
    port = _RecordingPort(20)
    optimizer = ColumnSerialOpenLoopSgd(
        port,
        device="cpu",
        binding_shapes=((4, 4), (2, 2)),
        layer_learning_rates_raw_x=(1.0, 1.0),
        nominal_delta_x=0.1,
        pulse_cap=1,
        pulse_selection_seed=9,
    )
    step = optimizer.step(
        (torch.ones((4, 4), dtype=torch.float32), torch.ones((2, 2), dtype=torch.float32))
    )
    # command=-lr*(2*dL/dG) has probability clipped to one everywhere.
    assert step.pulse.applied == 20
    direction, count = port.requests[-1]
    assert torch.equal(direction, -torch.ones(20, dtype=torch.int8))
    assert torch.equal(count, torch.ones(20, dtype=torch.int64))
    assert not hasattr(port, "verify")
    assert optimizer.state_dict()["verify_reads_during_updates"] == 0


def _tt_fixture(variant: str) -> tuple[OmPlantTikiTaka, _RecordingPort, IbmReramRawActivePlant]:
    shapes = ((4, 4), (2, 2))
    size = 20
    population = _population(size)
    fast = IbmReramRawActivePlant(
        population,
        seeds=tuple(range(101, 101 + size)),
        device="cpu",
    )
    symmetry = intrinsic_symmetry_raw_a(population)
    initialize_auxiliary_at_symmetry(fast, symmetry.raw_a)
    slow = _RecordingPort(size)
    optimizer = OmPlantTikiTaka(
        slow,
        fast,
        variant=variant,  # type: ignore[arg-type]
        symmetry_raw_a=symmetry.raw_a,
        device="cpu",
        binding_shapes=shapes,
        effective_transfer_lambdas=(0.2, 0.2),
        fast_learning_rate_raw_x=0.1,
        nominal_delta_x=0.1,
        pulse_cap=1,
        fast_pulse_selection_seed=31,
        slow_pulse_selection_seed=32,
    )
    # z_A=(apparent-a*)/2=0.5 on cursor-0 logical groups only.
    selected = torch.cat(
        (
            canonical_input_group_indices(shapes[0], 0, device="cpu"),
            canonical_input_group_indices(shapes[1], 0, device="cpu") + 16,
        )
    )
    fast.apparent[selected] = 1.0
    return optimizer, slow, fast


def test_tt_v1_reads_canonical_groups_and_uses_matched_effective_lambda() -> None:
    optimizer, slow, _fast = _tt_fixture("tt_v1")
    step = optimizer.step(
        (torch.zeros((4, 4), dtype=torch.float32), torch.zeros((2, 2), dtype=torch.float32))
    )
    # lambda*z_A=0.1=delta, so every one of 8+4 selected rail cells pulses.
    assert step.slow_c_pulse.applied == 12
    assert step.a_read_events == 2
    assert step.a_values_read == 12
    assert step.cursors_after == (1, 0)
    direction, count = slow.requests[-1]
    assert int(count.sum().item()) == 12
    assert bool(torch.all(direction[count.bool()] == 1))
    state = optimizer.state_dict()
    assert state["transfer_axis"] == "canonical_logical_[out,in]_input_columns"
    assert state["verify_reads_during_updates"] == 0
    assert "not_fresh_independent_read_noise" in state["a_read_state"]


def test_tt_v2_forgets_dispatched_h_but_retains_cap_blocked_debt() -> None:
    optimizer, slow, _fast = _tt_fixture("tt_v2")
    selected = torch.cat(
        (
            canonical_input_group_indices((4, 4), 0, device="cpu"),
            canonical_input_group_indices((2, 2), 0, device="cpu") + 16,
        )
    )
    assert optimizer.h is not None
    # Pre-cap one C cell. Its threshold crossing must remain as H debt.
    optimizer.slow_writer.pulse_count[selected[0]] = 1
    step = optimizer.step(
        (torch.zeros((4, 4), dtype=torch.float32), torch.zeros((2, 2), dtype=torch.float32))
    )
    assert step.h_threshold_crossings == (8, 4)
    assert step.slow_c_pulse.requested == 12
    assert step.slow_c_pulse.applied == 11
    assert step.slow_c_pulse.capped == 1
    assert optimizer.h[selected[0]].item() == 1.0
    assert torch.equal(optimizer.h[selected[1:]], torch.zeros(11))
    assert sum(step.h_cap_debt_retained) == 1
    direction, count = slow.requests[-1]
    assert count[selected[0]].item() == 0
    assert bool(torch.all(direction[count.bool()] == 1))
