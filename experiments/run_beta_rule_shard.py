"""Run an explicitly listed subset of the unchanged frozen beta measurements."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.list.read_text())
    sys.path.insert(0, config['runtime'])
    from experiments.reporting import atomic_write_json, sha256_file, utc_now, validate_run
    if sha256_file(__file__) != config['driver_sha256']:
        raise ValueError('Shard driver hash differs.')
    for path, expected in config['source_hashes'].items():
        if sha256_file(path) != expected:
            raise ValueError(f'Frozen source differs: {path}')
    root = Path(config['output'])
    state_path = root / config['progress_name']
    if state_path.exists():
        raise ValueError(f'Existing shard state requires reconciliation: {state_path}')
    origin = datetime.fromisoformat(config.get('budget_origin') or utc_now())
    def consumed():
        return ((datetime.now(timezone.utc)-origin).total_seconds()
                + config.get('prior_smoke_seconds', 0))
    state = dict(state='running', pid=os.getpid(), target=config['target'],
                 started_at=utc_now(), budget_origin=origin.isoformat(),
                 expected_cases=len(config['cases']), runs=[], current_case=None)
    def save():
        state.update(updated_at=utc_now(), charged_gpu_hours=consumed()/3600)
        atomic_write_json(state_path, state)
    save()
    try:
        for case in config['cases']:
            if sha256_file(case['config']) != case['sha256']:
                raise ValueError('Frozen config changed.')
            directory = root / 'production' / case['name']
            elapsed = None
            if not (directory / 'result.json').exists():
                if directory.exists():
                    raise ValueError(f'Incomplete case requires reconciliation: {directory}')
                remaining = config['cap_seconds']-consumed()
                if remaining < 180:
                    raise TimeoutError('Per-host GPU budget exhausted.')
                state['current_case'] = case['name']; save()
                command = [sys.executable, config['runner'], '--config', case['config'],
                           '--output-root', str(directory.parent), '--run-id', directory.name,
                           '--device', 'cuda', '--target', config['target']]
                logs = root / 'logs'; logs.mkdir(exist_ok=True)
                started = time.monotonic()
                print('START '+case['name'], flush=True)
                with (logs / ('shard_'+case['name']+'.log')).open('x') as stream:
                    process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                             timeout=min(900, remaining-30))
                elapsed = time.monotonic()-started
                if process.returncode:
                    raise RuntimeError(f'Case failed: {case["name"]}; preserve outputs for diagnosis.')
            if errors := validate_run(directory):
                raise ValueError(errors)
            manifest = json.loads((directory/'manifest.json').read_text())
            result = json.loads((directory/'result.json').read_text())
            m = result['terminal_metrics']
            if manifest['configuration']['sha256'] != case['sha256']:
                raise ValueError('Result config mismatch.')
            if (m['replay_count'] != 72 or m['layer_comparison_count'] != 72*(int(case['architecture'][-1])+1)
                or m['official_test_read'] or m['optimizer_steps_applied']
                or not m['source_bytes_unchanged'] or not m['all_bias_tensors_exact_zero']
                or m['endpoint_read_noise_std'] != 0):
                raise ValueError('Coverage or read-only guard failed.')
            with (directory/'whole_gradient_metrics.csv').open() as stream:
                if len(list(csv.DictReader(stream))) != 72:
                    raise ValueError('Whole-gradient coverage failed.')
            state['runs'].append(dict(case, state='complete', directory=str(directory),
                result_sha256=sha256_file(directory/'result.json'), elapsed_seconds=elapsed, **m))
            state['current_case']=None; save()
            print(f'DONE {len(state["runs"])}/{len(config["cases"])} {case["name"]} {elapsed}', flush=True)
        state.update(state='complete', completed_at=utc_now())
    except BaseException as error:
        state.update(state='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
