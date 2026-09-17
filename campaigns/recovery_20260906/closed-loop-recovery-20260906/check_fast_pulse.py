"""Require bitwise state/RNG parity and time the fused physical kernel."""
import json
import sys
import time
import torch
from run_recovery import HERE, Drn, cpu_tree
sys.path.insert(0, str(HERE / 'drn_code'))
from fast_drn_pulse import pulse

torch.set_num_threads(1)
plan = json.loads((HERE / 'plan.json').read_text())
adapter = Drn(plan, 'faulted', 1e-4, 64)
plant = adapter.plant
plant.pulse_count[:2048] = 639  # Canary-only cap-boundary exercise.
initial = cpu_tree(plant.state_dict())
generator = torch.Generator(device=plant.device).manual_seed(20260906)
directions = []
for round_index in range(32):
    draw = torch.rand(plant.raw_a.shape, generator=generator, device=plant.device)
    threshold = .45 if round_index % 4 == 0 else .001
    direction = (draw < threshold).to(torch.int8) - (draw > 1-threshold).to(torch.int8)
    directions.append(direction)
directions.append(torch.zeros_like(directions[0]))
# Warm compilation without advancing the comparison's state/RNG.
pulse(plant, directions[0], pulse_cap=640)
plant.load_state_dict(initial)
torch.cuda.synchronize()
start = time.perf_counter()
for direction in directions:
    plant.pulse(direction, pulse_cap=640)
torch.cuda.synchronize()
reference_seconds = time.perf_counter()-start
expected = cpu_tree(plant.state_dict())
plant.load_state_dict(initial)
start = time.perf_counter()
for direction in directions:
    pulse(plant, direction, pulse_cap=640)
torch.cuda.synchronize()
fused_seconds = time.perf_counter()-start
observed = cpu_tree(plant.state_dict())
checks = {}
for key in ('persistent_raw_a','apparent_raw_a','pulse_count','pulse_noise_rng_state','active_corrupt','active_stuck_raw_a'):
    checks[key] = torch.equal(expected[key], observed[key])
    if not checks[key]:
        print(key, 'max_difference', (expected[key].double()-observed[key].double()).abs().max().item())
assert all(checks.values()), checks
receipt = {'bitwise_parity': checks, 'rounds': len(directions), 'cap_and_empty_rounds_tested': True, 'cells': plant.size, 'reference_seconds': reference_seconds, 'fused_seconds': fused_seconds, 'speedup': reference_seconds/fused_seconds, 'CUDA': torch.cuda.get_device_name()}
(HERE/'smoke/fast-pulse-parity.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
