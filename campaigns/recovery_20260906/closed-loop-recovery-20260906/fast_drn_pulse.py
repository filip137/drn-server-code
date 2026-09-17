"""Fused physical pulse calculation; retain the original full-array RNG draws."""
import os
from pathlib import Path
os.environ.setdefault('TRITON_CACHE_DIR', str(Path(__file__).parent / 'triton_cache'))
import torch
import triton
import triton.language as tl


@triton.jit
def _pulse_kernel(raw_ptr, apparent_ptr, count_ptr, direction_ptr, up_ptr, down_ptr,
                  corrupt_ptr, stuck_ptr, cycle_ptr, write_ptr, N: tl.constexpr,
                  CAP: tl.constexpr, CYCLE_STD: tl.constexpr, WRITE_SCALE: tl.constexpr,
                  BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    raw = tl.load(raw_ptr+i, mask, other=0)
    apparent = tl.load(apparent_ptr+i, mask, other=0)
    count = tl.load(count_ptr+i, mask, other=0)
    direction = tl.load(direction_ptr+i, mask, other=0)
    up = tl.load(up_ptr+i, mask, other=0)
    down = tl.load(down_ptr+i, mask, other=0)
    corrupt = tl.load(corrupt_ptr+i, mask, other=0)
    stuck = tl.load(stuck_ptr+i, mask, other=0)
    cycle = tl.load(cycle_ptr+i, mask, other=0)
    write = tl.load(write_ptr+i, mask, other=0)
    selected = (direction != 0) & (count < CAP)
    candidate_up = raw + up * ((1.0 - raw) + CYCLE_STD * cycle)
    candidate_down = raw - down * ((1.0 + raw) + CYCLE_STD * cycle)
    candidate = tl.where(direction > 0, candidate_up, candidate_down)
    candidate = tl.minimum(tl.maximum(candidate, -1.0), 1.0)
    candidate = tl.where(corrupt, stuck, candidate)
    updated = tl.where(selected, candidate, raw)
    observed = tl.where(selected, updated + WRITE_SCALE * write, apparent)
    tl.store(raw_ptr+i, updated, mask)
    tl.store(apparent_ptr+i, observed, mask)
    tl.store(count_ptr+i, count + selected.to(tl.int64), mask)


def pulse(plant, direction, *, pulse_cap):
    from experiments.mnist_relu_drn.figure6_om_pulse import OM_CYCLE_NOISE_STD, OM_WRITE_NOISE_STD, OM_NOMINAL_DW_MIN_RAW_A
    selected = (direction != 0) & (plant.pulse_count < pulse_cap)
    if not bool(selected.any()):
        return
    # The original plant draws two complete float32 tensors per nonempty
    # pulse round. Preserve both shape and generator advancement exactly.
    cycle = torch.randn(plant.raw_a.shape, generator=plant.noise_generator, device=plant.device, dtype=plant.dtype)
    write = torch.randn(plant.raw_a.shape, generator=plant.noise_generator, device=plant.device, dtype=plant.dtype)
    _pulse_kernel[(triton.cdiv(plant.size, 256),)](
        plant.raw_a, plant.apparent_raw_a, plant.pulse_count, direction,
        plant.dwmin_up_raw_a, plant.dwmin_down_raw_a, plant.corrupt, plant.stuck_raw_a,
        cycle, write, plant.size, pulse_cap, OM_CYCLE_NOISE_STD,
        OM_WRITE_NOISE_STD * OM_NOMINAL_DW_MIN_RAW_A, 256, enable_fp_fusion=False)


@triton.jit
def _closed_round(raw_ptr, apparent_ptr, count_ptr, own_count_ptr, target_ptr,
                  up_ptr, down_ptr, corrupt_ptr, stuck_ptr, cycle_ptr, write_ptr,
                  stats_ptr, N: tl.constexpr, CAP: tl.constexpr, TOL: tl.constexpr,
                  CYCLE_STD: tl.constexpr, WRITE_SCALE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    raw = tl.load(raw_ptr+i, mask, other=0)
    apparent = tl.load(apparent_ptr+i, mask, other=0)
    count = tl.load(count_ptr+i, mask, other=0)
    own = tl.load(own_count_ptr+i, mask, other=0)
    target = tl.load(target_ptr+i, mask, other=0)
    # Controller decisions use only target, apparent observation, own history.
    error = target - ((apparent + 1.0) * 0.5)
    selected = (tl.abs(error) > TOL) & (own < CAP) & mask
    up = tl.load(up_ptr+i, mask, other=0)
    down = tl.load(down_ptr+i, mask, other=0)
    corrupt = tl.load(corrupt_ptr+i, mask, other=0)
    stuck = tl.load(stuck_ptr+i, mask, other=0)
    cycle = tl.load(cycle_ptr+i, mask, other=0)
    write = tl.load(write_ptr+i, mask, other=0)
    candidate_up = raw + up * ((1.0 - raw) + CYCLE_STD * cycle)
    candidate_down = raw - down * ((1.0 + raw) + CYCLE_STD * cycle)
    candidate = tl.where(error > 0, candidate_up, candidate_down)
    candidate = tl.minimum(tl.maximum(candidate, -1.0), 1.0)
    candidate = tl.where(corrupt, stuck, candidate)
    updated = tl.where(selected, candidate, raw)
    observed = tl.where(selected, updated + WRITE_SCALE * write, apparent)
    own_new = own + selected.to(tl.int64)
    tl.store(raw_ptr+i, updated, mask)
    tl.store(apparent_ptr+i, observed, mask)
    tl.store(count_ptr+i, count + selected.to(tl.int64), mask)
    tl.store(own_count_ptr+i, own_new, mask)
    remaining_error = target - ((observed + 1.0) * 0.5)
    unresolved = (tl.abs(remaining_error) > TOL) & mask
    blocked = unresolved & (own_new >= CAP)
    tl.atomic_add(stats_ptr, tl.sum(selected.to(tl.int32)))
    tl.atomic_add(stats_ptr+1, tl.sum(unresolved.to(tl.int32)))
    tl.atomic_add(stats_ptr+2, tl.sum(blocked.to(tl.int32)))


@torch.no_grad()
def closed_step(self, gradient):
    """Same Adam arithmetic, P&V decisions and RNG; fuse intermediate tensors."""
    from experiments.mnist_relu_drn.figure6_om_pulse import OM_CYCLE_NOISE_STD, OM_WRITE_NOISE_STD, OM_NOMINAL_DW_MIN_RAW_A
    plant = self.read_apparent.__self__.plant
    value = gradient.detach().reshape(-1)
    if value.shape != self.target.shape or not bool(torch.isfinite(value).all()):
        raise ValueError('Expected one finite gradient per device.')
    self.step_index += 1
    self.first_moment.mul_(0.9).add_(value, alpha=0.1)
    self.second_moment.mul_(0.999).addcmul_(value, value, value=0.001)
    first = self.first_moment / (1 - 0.9 ** self.step_index)
    second = self.second_moment / (1 - 0.999 ** self.step_index)
    command = -self.learning_rate * first / (second.sqrt() + self.epsilon)
    self.target.add_(command)
    error = self.target - self.read_apparent()
    unresolved_mask = error.abs() > self.tolerance
    blocked_mask = unresolved_mask & (self.pulse_count >= self.total_pulse_cap)
    unresolved, blocked = int(unresolved_mask.sum()), int(blocked_mask.sum())
    self.verify_reads += self.target.numel()
    pulses, rounds = 0, 0
    while unresolved > blocked and rounds < self.maximum_pulses:
        cycle = torch.randn(plant.raw_a.shape, generator=plant.noise_generator, device=plant.device, dtype=plant.dtype)
        write = torch.randn(plant.raw_a.shape, generator=plant.noise_generator, device=plant.device, dtype=plant.dtype)
        stats = torch.zeros(3, dtype=torch.int64, device=plant.device)
        _closed_round[(triton.cdiv(plant.size, 256),)](
            plant.raw_a, plant.apparent_raw_a, plant.pulse_count, self.pulse_count, self.target,
            plant.dwmin_up_raw_a, plant.dwmin_down_raw_a, plant.corrupt, plant.stuck_raw_a,
            cycle, write, stats, plant.size, self.total_pulse_cap, self.tolerance,
            OM_CYCLE_NOISE_STD, OM_WRITE_NOISE_STD * OM_NOMINAL_DW_MIN_RAW_A, 256, enable_fp_fusion=False)
        count, unresolved, blocked = stats.tolist()
        pulses += count
        rounds += 1
        self.verify_reads += count
    self.cap_blocks += blocked
    if rounds == self.maximum_pulses:
        self.burst_exhaustions += unresolved - blocked
    return {'pulses': pulses, 'verify_rounds': rounds, 'unresolved_cells': unresolved, 'blocked_at_cap': blocked}
