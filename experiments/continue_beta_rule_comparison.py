"""Continue the unchanged beta grid with a two-worker local transport benchmark."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import os
from pathlib import Path
import time

from experiments.reporting import atomic_write_json, sha256_file, utc_now
from experiments.run_beta_rule_comparison import (
    OUT, ROOT, CAP_SECONDS, checked, read, run_case, verify_inputs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('benchmark', 'remainder'), required=True)
    args = parser.parse_args()
    verify_inputs()
    inputs_path = OUT / 'parallel_transport.json'
    inputs = read(inputs_path)
    if inputs['driver_sha256'] != sha256_file(__file__):
        raise ValueError('Frozen transport driver changed.')
    old = read(OUT / 'execution.single_worker.json')
    try:
        os.kill(old['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise RuntimeError('Original driver is still alive; refuse concurrent admission.')
    cases = read(OUT / 'cases.json')
    completed = []
    pending = []
    for case in cases:
        directory = OUT / 'production' / case['name']
        if (directory / 'result.json').is_file():
            completed.append(dict(checked(case, directory), elapsed_seconds=None))
        elif directory.exists():
            raise ValueError(f'Incomplete existing case requires reconciliation: {directory}')
        else:
            pending.append(case)
    if args.stage == 'benchmark':
        if (OUT / 'parallel_benchmark.json').exists():
            raise ValueError('Benchmark already exists.')
        admitted = pending[:2]
        workers = 2
    else:
        benchmark = read(OUT / 'parallel_benchmark.json')
        admitted = pending
        workers = benchmark['selected_workers']
    origin = datetime.fromisoformat(old['started_at'])
    def consumed():
        return (datetime.now(timezone.utc) - origin).total_seconds() + old['smoke_seconds']
    summary = dict(old, state='running', pid=os.getpid(), runs=completed,
        current_case=None, current_cases=[c['name'] for c in admitted[:workers]],
        transport_stage=args.stage, workers=workers, updated_at=utc_now())
    summary.pop('error', None)
    atomic_write_json(OUT / 'execution.json', summary)
    started = time.monotonic()
    def work(case):
        remaining = CAP_SECONDS - consumed()
        if remaining < 180:
            raise TimeoutError('Original six-GPU-hour cap is exhausted.')
        print('START '+case['name'], flush=True)
        return run_case(case, OUT / 'production' / case['name'], min(900, remaining-30))
    try:
        # Only workers cases are admitted at a time; errors cannot submit a hidden backlog.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            iterator = iter(admitted)
            live = {}
            for _ in range(workers):
                if case := next(iterator, None): live[pool.submit(work, case)] = case
            while live:
                future = next(as_completed(live))
                case = live.pop(future)
                result = future.result()
                if result['state'] != 'complete':
                    raise RuntimeError('Numerical failure requires explicit retained-case reconciliation.')
                summary['runs'].append(result)
                if following := next(iterator, None):
                    live[pool.submit(work, following)] = following
                summary.update(current_cases=[c['name'] for c in live.values()],
                    elapsed_seconds=consumed()-old['smoke_seconds'], updated_at=utc_now())
                atomic_write_json(OUT / 'execution.json', summary)
                print(f'DONE {len(summary["runs"])}/153 {case["name"]} {result["elapsed_seconds"]:.1f}s',flush=True)
        if args.stage == 'benchmark':
            elapsed = time.monotonic() - started
            old_times = [r['elapsed_seconds'] for r in old['runs'] if r['architecture']=='conv3']
            reference_seconds = 2 * sum(old_times)/len(old_times)
            speedup = reference_seconds/elapsed
            # Keep concurrency only with a material measured throughput benefit.
            selected = 2 if speedup >= 1.10 else 1
            atomic_write_json(OUT / 'parallel_benchmark.json', dict(elapsed_seconds=elapsed,
                sequential_reference_seconds=reference_seconds, speedup=speedup,
                selected_workers=selected, cases=[c['name'] for c in admitted],
                unchanged_scientific_configs=True, peak_memory_record='parallel_gpu_samples.csv'))
            summary.update(state='benchmark_complete', current_cases=[], selected_workers=selected)
        else:
            if len(summary['runs']) != 153:
                raise ValueError('Coverage mismatch.')
            summary.update(state='complete', current_case=None, current_cases=[], completed_at=utc_now())
    except BaseException as error:
        summary.update(state='failed', error=repr(error))
        raise
    finally:
        summary.update(elapsed_seconds=consumed()-old['smoke_seconds'], charged_gpu_hours=consumed()/3600)
        atomic_write_json(OUT / 'execution.json', summary)


if __name__ == '__main__':
    main()
