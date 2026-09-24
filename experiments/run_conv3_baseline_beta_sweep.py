"""Sweep wide Conv3 baseline beta at fixed T8/K8; confirm one frozen candidate."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.reporting import validate_run
from experiments.run_baseline_wide_t_audit import read, sha, write

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'eqprop-conv3-baseline-beta-tk8-20260916-v1'
OUT = ROOT / 'results' / STUDY
PRIOR = ROOT / 'results/eqprop-beta-selection-audit-20260914-v1'
T_STUDY = ROOT / 'results/eqprop-baseline-wide-t-relaxation-20260916-v1'
OLD_CONFIGS = ROOT / 'configs/conv/eqprop_beta_selection_audit_20260914_v1/v2'
CONFIGS = ROOT / 'configs/conv/eqprop_conv3_baseline_beta_tk8_20260916_v1'
PLAN = ROOT / 'docs/eqprop_conv3_baseline_beta_tk8_plan_20260916.md'
RUNNER = OUT / 'source/experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py'
BETAS = (10, 30, 50, 75)


def reference():
    row = read(OUT / 'reference.json')
    directory = (ROOT / row['result']).parent
    if errors := validate_run(directory):
        raise ValueError(errors)
    if sha(directory / 'result.json') != row['result_sha256']:
        raise ValueError('Reused beta100 result changed.')
    return dict(row, phase='reference', seed=0, **read(directory / 'result.json')['terminal_metrics'])


def prepare():
    if STUDY not in (ROOT / 'docs/current_simulations.md').read_text():
        raise ValueError('Register the persistent study row before preparation.')
    prior_execution = read(T_STUDY / 'execution.json')
    if (prior_execution['state'] != 'complete' or prior_execution['selected_T'] is not None
            or any(r['phase'] == 'confirmation' for r in prior_execution['runs'])
            or (T_STUDY / 'confirmation').exists()):
        raise ValueError('Reserved confirmation cohort is no longer unmeasured.')
    frozen = read(PRIOR / 'source_identity.json')
    for name, expected in frozen.items():
        source, dest = PRIOR / 'source' / name, OUT / 'source' / name
        if sha(source) != expected or (dest.exists() and sha(dest) != expected):
            raise ValueError('Frozen analyzer changed.')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    write(OUT / 'source_identity.json', frozen, frozen=True)
    for name in ('partition.json', 'confirmation.json'):
        write(OUT / 'cohorts' / name, read(T_STUDY / 'cohorts' / name), frozen=True)
    prior_case = ROOT / 'configs/conv/eqprop_baseline_wide_t_20260916_v1/confirmation_conv3_baseline_seed0_T12_K8_beta100.json'
    confirmation = read(prior_case)['dataset']
    prepared = []
    for phase in ('selection', 'confirmation'):
        for seed in ([0] if phase == 'selection' else [0, 1, 2]):
            template = read(OLD_CONFIGS / f'{phase}_conv3_baseline_seed{seed}_beta100.json')
            for beta in BETAS:
                config = copy.deepcopy(template)
                config.update(study_id=STUDY,
                    scientific_question='Which tested beta passes expanded Conv3 baseline gradient checks at the user-fixed T8/K8, retaining the separate equilibrium caveat?')
                case = config['cases'][0]
                assert case['T'] == case['K'] == 8 and 'replay_T' not in case
                case.update(base_beta=beta, injected_beta=beta)
                if phase == 'confirmation':
                    config['dataset'] = confirmation
                name = f'{phase}_conv3_baseline_seed{seed}_T8_K8_beta{beta}'
                path = CONFIGS / f'{name}.json'
                write(path, config, frozen=True)
                prepared.append(dict(name=name, phase=phase, seed=seed, T=8, K=8,
                    beta=beta, path=str(path.relative_to(ROOT)), sha256=sha(path)))
    write(OUT / 'prepared_configs.json', prepared, frozen=True)
    old_ref = read(T_STUDY / 'reference.json')
    write(OUT / 'reference.json', old_ref, frozen=True)
    reference()
    frozen_plan = OUT / 'plan.frozen.md'
    if frozen_plan.exists() and sha(frozen_plan) != sha(PLAN):
        raise ValueError('Frozen plan differs.')
    shutil.copy2(PLAN, frozen_plan)
    write(OUT / 'execution_inputs.json', dict(driver_sha256=sha(__file__),
        helper_sha256=sha(ROOT / 'experiments/run_baseline_wide_t_audit.py'),
        plan_sha256=sha(frozen_plan), cap_gpu_hours=1, selection_cases_new=4,
        confirmation_cases_maximum=3, retained_T8_equilibrium_caveat=True,
        unused_confirmation_proof_sha256=sha(T_STUDY / 'execution.json')), frozen=True)
    print(f'Prepared {len(prepared)} configs; fixed T8/K8; fresh confirmation cohort retained.')


def checked(case, *, smoke=False):
    directory = OUT / ('smoke' if smoke else case['phase']) / case['name']
    if sha(ROOT / case['path']) != case['sha256']:
        raise ValueError('Config changed.')
    if errors := validate_run(directory):
        raise ValueError(f'{directory}: {errors}')
    manifest = read(directory / 'manifest.json')
    if manifest['configuration']['sha256'] != case['sha256']:
        raise ValueError('Result/config mismatch.')
    metrics = read(directory / 'result.json')['terminal_metrics']
    count = 1 if smoke else 72
    if metrics['replay_count'] != count or metrics['layer_comparison_count'] != count*4:
        raise ValueError('Incomplete coverage.')
    if (metrics['official_test_read'] or metrics['optimizer_steps_applied']
            or not metrics['source_bytes_unchanged'] or not metrics['all_bias_tensors_exact_zero']
            or metrics['endpoint_read_noise_std'] != 0):
        raise ValueError('Read-only clean replay guards failed.')
    return dict(case, result=str((directory / 'result.json').relative_to(ROOT)),
                result_sha256=sha(directory / 'result.json'), **metrics)


def run():
    if (OUT / 'execution.json').exists():
        raise ValueError('Existing execution needs explicit reconciliation.')
    inputs = read(OUT / 'execution_inputs.json')
    if (inputs['driver_sha256'] != sha(__file__)
            or inputs['helper_sha256'] != sha(ROOT / 'experiments/run_baseline_wide_t_audit.py')):
        raise ValueError('Prepared driver/helper changed.')
    for name, expected in read(OUT / 'source_identity.json').items():
        if sha(OUT / 'source' / name) != expected:
            raise ValueError('Frozen analyzer changed.')
    cases = read(OUT / 'prepared_configs.json')
    ref = reference()
    started = time.monotonic()
    summary = dict(study_id=STUDY, state='running', pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(), target='main:RTX3090',
        cap_gpu_hours=1, runs=[], reference=ref, selected_beta=None,
        retained_T8_equilibrium_caveat=True, official_test_read=False, training_started=False)

    def run_case(case, *, smoke=False):
        phase = 'smoke' if smoke else case['phase']
        directory = OUT / phase / case['name']
        remaining = 3600 - (time.monotonic() - started)
        if directory.exists() or remaining < 180:
            raise ValueError('Existing output or insufficient remaining study budget.')
        command = [sys.executable, str(RUNNER), '--config', str(ROOT / case['path']),
            '--output-root', str(directory.parent), '--run-id', case['name'],
            '--device', 'cuda', '--target', 'main:RTX3090']
        if smoke:
            command.append('--smoke')
        summary.update(current_case=case['name'], current_phase=phase,
                       elapsed_seconds=time.monotonic()-started)
        write(OUT / 'execution.json', summary)
        print(f'START {phase} beta={case["beta"]} seed={case["seed"]}', flush=True)
        with (OUT / f'{phase}_{case["name"]}.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                check=True, timeout=min(180 if smoke else 600, remaining-30))
        result = checked(case, smoke=smoke)
        if smoke:
            summary['smoke'] = result
        else:
            summary['runs'].append(result)
        summary['elapsed_seconds'] = time.monotonic()-started
        write(OUT / 'execution.json', summary)
        print(f'DONE {phase} beta={case["beta"]} seed={case["seed"]}: '
            f'cosine={result["minimum_cosine"]:.6f} norm={result["maximum_symmetric_norm_delta"]:.6f} '
            f'gradient={result["gradient_fidelity_all_passed"]} equilibrium={result["equilibrium_residual_all_passed"]}', flush=True)
        return result

    try:
        run_case(cases[0], smoke=True)
        for case in cases:
            if case['phase'] == 'selection':
                run_case(case)
        passing = [r for r in [ref, *summary['runs']] if r['gradient_fidelity_all_passed']]
        selected = max((r['beta'] for r in passing), default=None)
        decision = dict(selected_beta=selected, T=8, K=8, frozen_before_confirmation=True,
            state='confirmation_pending' if selected is not None else 'no_passing_beta',
            criterion='all_rows_gradient_fidelity; equilibrium reported separately under user-fixed T8 caveat',
            selection_results=[dict(result=r['result'], sha256=r['result_sha256'])
                               for r in [ref, *summary['runs']]])
        write(OUT / 'selection_decision.json', decision, frozen=True)
        summary.update(selected_beta=selected, decision=decision['state'])
        write(OUT / 'execution.json', summary)
        if selected is not None:
            confirmations = [run_case(c) for c in cases
                             if c['phase'] == 'confirmation' and c['beta'] == selected]
            assert len(confirmations) == 3
            summary['decision'] = ('gradient_confirmed_T8_equilibrium_caveat'
                if all(r['gradient_fidelity_all_passed'] for r in confirmations) else 'confirmation_failed')
        summary.update(state='complete', current_case=None, current_phase=None,
                       completed_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        summary.update(state='failed', error=repr(exc))
        raise
    finally:
        summary['elapsed_seconds'] = time.monotonic()-started
        summary['charged_gpu_hours'] = summary['elapsed_seconds']/3600
        write(OUT / 'execution.json', summary)
        (OUT / 'gpu_hours.csv').write_text('study,cap_gpu_hours,charged_gpu_hours,state\n'
            f'{STUDY},1,{summary["charged_gpu_hours"]:.12f},{summary["state"]}\n')
        print({k:summary.get(k) for k in ('state','decision','selected_beta','charged_gpu_hours')}, flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('prepare', 'run'):
        raise SystemExit('Usage: python -m experiments.run_conv3_baseline_beta_sweep prepare|run')
    (prepare if sys.argv[1] == 'prepare' else run)()
