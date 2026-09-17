"""Operational acceleration of DRN P&V with a bitwise-equivalent pulse plant."""
import argparse
import json
from pathlib import Path
from run_recovery import HERE, Drn, file_hash, run, write_json
from fast_drn_pulse import pulse, closed_step
from closed_loop import ClosedLoopAdam


def accelerated_pulse(self, direction):
    pulse(self.plant, direction, pulse_cap=self.open_optimizer.pulse_cap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=['healthy', 'faulted'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--maximum-batches', type=int, default=3438)
    parser.add_argument('--evaluation-limit', type=int)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    args.architecture, args.writer, args.plan = 'drn', 'closed_loop_pv', HERE / 'plan.json'
    parity = json.loads((HERE / 'smoke/fast-pulse-parity.json').read_text())
    if not all(parity['bitwise_parity'].values()):
        raise RuntimeError('Expected bitwise physical-kernel parity before acceleration.')
    full_parity = json.loads((HERE / 'smoke/full-controller-parity.json').read_text())
    if not all(full_parity['bitwise_parity'].values()):
        raise RuntimeError('Expected exact full-controller replay before acceleration.')
    receipt = {'intervention': 'operational_kernel_fusion_only', 'persistent_and_apparent_equations_unchanged': True, 'full_array_RNG_draws_and_generator_advancement_unchanged': True, 'floating_point_fusion_disabled': True, 'files': {name: file_hash(HERE / name) for name in ('run_fast_recovery.py', 'fast_drn_pulse.py', 'run_recovery.py', 'closed_loop.py', 'smoke/fast-pulse-parity.json', 'smoke/full-controller-parity.json')}}
    Drn.pulse = accelerated_pulse
    ClosedLoopAdam.step = closed_step
    try:
        run(args)
    finally:
        if args.output.is_dir():
            write_json(args.output / 'acceleration.json', receipt)
