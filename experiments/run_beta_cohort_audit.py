"""Run the declared baseline beta grid, then confirm only a frozen candidate."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.reporting import validate_run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/eqprop-beta-selection-audit-20260914-v1'
RUNNER = OUT / 'source/experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def validated_result(case):
    if sha(ROOT / case['path']) != case['sha256']:
        raise ValueError('Frozen audit config changed.')
    directory = OUT / case['phase'] / case['name']
    errors = validate_run(directory)
    if errors:
        raise ValueError(f'{directory}: {errors}')
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['configuration']['sha256'] != case['sha256']:
        raise ValueError('Existing result uses another config.')
    result = json.loads((directory / 'result.json').read_text())
    terminal = result['terminal_metrics']
    if terminal['replay_count'] != 72 or terminal['layer_comparison_count'] != 72*(int(case['architecture'][-1])+1):
        raise ValueError('Incomplete replay coverage.')
    if (terminal['official_test_read'] or terminal['optimizer_steps_applied']
            or not terminal['source_bytes_unchanged'] or not terminal['all_bias_tensors_exact_zero']):
        raise ValueError('Read-only replay guards failed.')
    return dict(case, result=str((directory / 'result.json').relative_to(ROOT)),
                result_sha256=sha(directory / 'result.json'), **terminal)


def main():
    started = time.monotonic()
    # Separate first-case and smoke limits leave this 90-minute wave inside
    # the already reserved two-hour audit cap.
    deadline = started + 5400
    cases = json.loads((OUT / 'prepared_configs_v2.json').read_text())
    for name, expected in json.loads((OUT / 'source_identity.json').read_text()).items():
        if sha(OUT / 'source' / name) != expected:
            raise ValueError('Frozen analyzer changed.')
    summary = dict(study_id=OUT.name, state='running', pid=os.getpid(),
                   started_at=datetime.now(timezone.utc).isoformat(),
                   selection_expected=9, confirmation_maximum=9,
                   official_test_read=False, training_started=False, runs=[], decisions={})

    def run_case(case):
        directory = OUT / case['phase'] / case['name']
        if not (directory / 'result.json').is_file():
            if directory.exists():
                raise ValueError(f'Incomplete existing run needs explicit recovery: {directory}')
            remaining = deadline - time.monotonic()
            if remaining < 120:
                raise TimeoutError('Audit wall budget exhausted before next case.')
            command = [sys.executable, str(RUNNER), '--config', str(ROOT / case['path']),
                       '--output-root', str(directory.parent), '--run-id', case['name'],
                       '--device', 'cuda', '--target', 'main']
            summary['current_case'] = case['name']
            write(OUT / 'execution.json', summary)
            print('START ' + case['name'], flush=True)
            with (OUT / f"{case['name']}.log").open('x') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=min(600, remaining - 30))
        result = validated_result(case)
        summary['runs'].append(result)
        summary['elapsed_seconds'] = time.monotonic() - started
        write(OUT / 'execution.json', summary)
        print(f"DONE {case['name']}: cosine={result['minimum_cosine']:.6f} "
              f"gradient_pass={result['gradient_fidelity_all_passed']} "
              f"residual_pass={result['equilibrium_residual_all_passed']}", flush=True)
        return result

    try:
        # Finish the full grid before looking at any confirmation gradients.
        for case in cases:
            if case['phase'] == 'selection':
                run_case(case)
        for architecture in ('conv3', 'conv2', 'conv1'):
            rows = [r for r in summary['runs'] if r['architecture'] == architecture and r['phase'] == 'selection']
            passing = [r for r in rows if r['gradient_fidelity_all_passed']]
            selected = max(passing, key=lambda r: r['beta']) if passing else None
            decision = dict(architecture=architecture,
                            selected_beta=selected['beta'] if selected else None,
                            selection_equilibrium_passed=selected['equilibrium_residual_all_passed'] if selected else False,
                            selected_before_confirmation=True,
                            state='confirmation_pending' if selected else 'no_passing_tested_beta',
                            selection_results=[{'path':r['result'], 'sha256':r['result_sha256']} for r in rows])
            # Frozen choice; a confirmation failure cannot select another beta.
            target = OUT / 'decisions' / f'{architecture}_selection.json'
            if target.exists():
                raise ValueError('A frozen decision already exists; reconcile before resuming.')
            write(target, decision)
            summary['decisions'][architecture] = decision
        write(OUT / 'execution.json', summary)
        for architecture, decision in summary['decisions'].items():
            if decision['selected_beta'] is None:
                continue
            confirmations = []
            for case in cases:
                if case['phase'] == 'confirmation' and case['architecture'] == architecture and case['beta'] == decision['selected_beta']:
                    confirmations.append(run_case(case))
            assert len(confirmations) == 3
            gradient = all(r['gradient_fidelity_all_passed'] for r in confirmations)
            equilibrium = decision['selection_equilibrium_passed'] and all(r['equilibrium_residual_all_passed'] for r in confirmations)
            decision.update(state='confirmed' if gradient and equilibrium else
                            'gradient_confirmed_equilibrium_held' if gradient else 'confirmation_failed',
                            confirmation_gradient_passed=gradient,
                            confirmation_equilibrium_passed=equilibrium)
        summary.update(state='complete', current_case=None,
                       completed_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        summary.update(state='failed', error=repr(exc))
        raise
    finally:
        summary['elapsed_seconds'] = time.monotonic() - started
        write(OUT / 'execution.json', summary)


if __name__ == '__main__':
    main()
