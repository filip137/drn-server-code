"""Exercise writes and faulted-state replay on each real CUDA plant."""
import json
from pathlib import Path
import sys
import torch
from run_recovery import HERE, Drn, Crossbar, cpu_tree, tensor_hash
from closed_loop import ClosedLoopAdam

architecture = sys.argv[1]
sys.path.insert(0, str(HERE / f'{architecture}_code'))
torch.set_num_threads(1)
plan = json.loads((HERE / 'plan.json').read_text())
adapter = (Drn if architecture == 'drn' else Crossbar)(plan, 'faulted', 1e-4, 64)
state = cpu_tree(adapter.state())
before_a, before_p = adapter.apparent().clone(), adapter.persistent().clone()
optimizer = ClosedLoopAdam(adapter.apparent, adapter.pulse, learning_rate=.1, tolerance=adapter.tolerance)
gradient = torch.zeros_like(before_a)
gradient[:64] = -1
result = optimizer.step(gradient)
assert result['pulses'] > 0
assert torch.equal(before_a[64:], adapter.apparent()[64:])
assert torch.equal(before_p[64:], adapter.persistent()[64:])
invariants = adapter.invariant()
assert invariants['bounds_respected'] and invariants['faults_immutable']
adapter.restore(state)
assert torch.equal(before_a, adapter.apparent()) and torch.equal(before_p, adapter.persistent())
receipt = {'architecture': architecture, 'CUDA': torch.cuda.get_device_name(), 'forced_command_canary_only': True, 'program_verify': result, 'untouched_cells_bitwise_unchanged': True, 'checkpoint_replay_exact': True, 'invariants': invariants}
(HERE / 'smoke' / f'{architecture}-physical-check.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
