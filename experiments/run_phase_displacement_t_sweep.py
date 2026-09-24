"""Replay only the affected displacement cases while varying free-phase T."""
from __future__ import annotations

import copy
import csv
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
STUDY = 'eqprop-phase-displacement-t-sweep-20260916-v1'
OUT = ROOT / 'results' / STUDY
PRIOR = ROOT / 'results/eqprop-phase-displacement-conv123-20260916-v1'
CONFIGS = ROOT / 'configs/conv/eqprop_phase_displacement_t_sweep_20260916_v1'
PLAN = ROOT / 'docs/eqprop_phase_displacement_t_sweep_plan_20260916.md'
RUNNER = OUT / 'source/experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py'
ROLES = ['reconstructed_initialization', 'best_validation']
SURFACES = [('conv3', 'baseline', [8, 10, 12, 16, 24], ROLES),
            ('conv3', 'legacy', [8, 10, 12, 16, 24], ROLES),
            ('conv2', 'legacy', [6, 8, 10, 12, 16], ROLES[:1])]
CONTEXTS = {('zero', 'post_T_free'), ('positive', 'post_T_free'),
            ('negative', 'post_T_free'), ('positive', 'matched_zero_K'),
            ('negative', 'matched_zero_K'), ('positive_minus_negative', 'negative_phase')}


def completion_counts(config):
    """Derive semantic coverage from the requested cases, roles and cohort."""
    batches = len(config['dataset']['expected_batch_guards'])
    roles = len(config['checkpoint_roles'])
    return dict(production_replay_count=batches * roles * len(config['cases']),
                production_layer_comparison_count=batches * roles * sum(
                    int(c['architecture'][-1]) + 1 for c in config['cases']))


def prepare():
    if STUDY not in (ROOT / 'docs/current_simulations.md').read_text():
        raise ValueError('Register the persistent study row before preparation.')
    snapshot = {}
    changed = {'experiments/audit_eqprop_float64_shadow.py',
               'experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py'}
    for name, digest in read(PRIOR / 'source_identity.json').items():
        previous = PRIOR / 'source' / name
        if sha(previous) != digest:
            raise ValueError('Previous numerical snapshot changed.')
        source = ROOT / name if name in changed else previous
        dest = OUT / 'source' / name
        if dest.exists() and sha(dest) != sha(source):
            raise ValueError('Existing snapshot differs.')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        snapshot[name] = sha(dest)
    write(OUT / 'source_identity.json', snapshot, frozen=True)
    cases = []
    for wave in range(5):
        for architecture, scheme, ts, roles in SURFACES:
            prior_name = f'{architecture}_{scheme}_seed0'
            template = read(ROOT / f'configs/conv/eqprop_phase_displacement_20260916_v1/{prior_name}.json')
            config = copy.deepcopy(template)
            config.update(study_id=STUDY, checkpoint_roles=roles,
                record_zero_nudge_displacement=True,
                scientific_question='How does increasing only T change residual relaxation versus nudging displacement in the affected seed0 checkpoints?')
            source = config['cases'][0]
            source.update(replay_T=ts[wave], replay_K=source['K'])
            config['completion'].update(completion_counts(config))
            name = f'{architecture}_{scheme}_seed0_T{ts[wave]}'
            path = CONFIGS / f'{name}.json'
            write(path, config, frozen=True)
            previous = PRIOR / 'runs' / prior_name / 'result.json'
            cases.append(dict(name=name, architecture=architecture, scheme=scheme,
                T=ts[wave], K=source['K'], native_T=source['T'], beta=source['injected_beta'],
                base_beta=source['base_beta'], roles=roles, seed=0,
                path=str(path.relative_to(ROOT)), sha256=sha(path),
                previous_result=str(previous.relative_to(ROOT)), previous_result_sha256=sha(previous)))
    write(OUT / 'prepared_configs.json', cases, frozen=True)
    shutil.copy2(PLAN, OUT / 'plan.frozen.md')
    write(OUT / 'execution_inputs.json', dict(driver_sha256=sha(__file__),
        helper_sha256=sha(ROOT / 'experiments/run_baseline_wide_t_audit.py'),
        plan_sha256=sha(OUT / 'plan.frozen.md'), cap_gpu_hours=1,
        formal_cases=15, expected_replays=900, expected_gradient_comparisons=3420,
        expected_individual_layer_rows=20520, expected_pooled_contexts=570), frozen=True)
    print('Prepared15 T-only cases;900 replays with direct zero-nudge drift.')


def checked(case, *, smoke=False):
    directory = OUT / ('smoke' if smoke else 'runs') / case['name']
    if sha(ROOT / case['path']) != case['sha256']:
        raise ValueError('Frozen config changed.')
    if errors := validate_run(directory):
        raise ValueError(f'{directory}: {errors}')
    manifest = read(directory / 'manifest.json')
    if manifest['configuration']['sha256'] != case['sha256']:
        raise ValueError('Configuration identity mismatch.')
    m = read(directory / 'result.json')['terminal_metrics']
    replays = 1 if smoke else 36 * len(case['roles'])
    depth = int(case['architecture'][-1])
    if m['replay_count'] != replays or m['layer_comparison_count'] != replays * (depth + 1):
        raise ValueError('Incomplete replay coverage.')
    if (m['official_test_read'] or m['optimizer_steps_applied'] or m['endpoint_read_noise_std'] != 0
            or not m['source_bytes_unchanged'] or not m['all_bias_tensors_exact_zero']):
        raise ValueError('Clean read-only guards failed.')
    rows = list(csv.DictReader((directory / 'state_displacement.csv').open()))
    if len(rows) != replays * (depth + 2) * 6:
        raise ValueError('Incomplete six-context displacement coverage.')
    if {(r['phase'], r['reference_kind']) for r in rows} != CONTEXTS:
        raise ValueError('Missing displacement context.')
    if any(int(r['T']) != case['T'] or int(r['K']) != case['K'] for r in rows):
        raise ValueError('T-only override did not take effect.')
    if {r['checkpoint_role'] for r in rows} != ({'best_validation'} if smoke else set(case['roles'])):
        raise ValueError('Wrong checkpoint coverage.')
    return dict(case, result=str((directory / 'result.json').relative_to(ROOT)),
                result_sha256=sha(directory / 'result.json'), **m)


def regression(case, *, smoke=False):
    if case['T'] != case['native_T']:
        return None
    old = PRIOR / 'smoke/conv3_baseline_seed0' if smoke else (ROOT / case['previous_result']).parent
    new = OUT / ('smoke' if smoke else 'runs') / case['name']
    if not smoke and sha(old / 'result.json') != case['previous_result_sha256']:
        raise ValueError('Prior reference changed.')
    roles = {'best_validation'} if smoke else set(case['roles'])
    for name in ('layer_metrics.csv', 'state_displacement.csv', 'equilibrium_residuals.csv'):
        before = [r for r in csv.DictReader((old / name).open()) if r['checkpoint_role'] in roles]
        after = [r for r in csv.DictReader((new / name).open()) if r['checkpoint_role'] in roles
                 and not (name == 'state_displacement.csv' and r['phase'] == 'zero')]
        if len(before) != len(after) or any(a != b for a, b in zip(before, after, strict=True)):
            raise ValueError(f'Native-T shared measurements differ: {case["name"]}/{name}')
    proof = dict(case=case['name'], smoke=smoke, shared_measurements_identical=True,
                 previous_result=str((old / 'result.json').relative_to(ROOT)),
                 previous_result_sha256=sha(old / 'result.json'))
    write(OUT / ('smoke_regression.json' if smoke else f'regression_{case["name"]}.json'), proof)
    return proof


def run(*, resume=False):
    if not resume and (OUT / 'execution.json').exists():
        raise ValueError('Existing execution requires reconciliation.')
    inputs = read(OUT / ('recovery_inputs.json' if resume else 'execution_inputs.json'))
    if inputs['driver_sha256'] != sha(__file__) or inputs['helper_sha256'] != sha(ROOT / 'experiments/run_baseline_wide_t_audit.py'):
        raise ValueError('Frozen driver/helper changed.')
    for name, digest in read(OUT / 'source_identity.json').items():
        if sha(OUT / 'source' / name) != digest:
            raise ValueError('Frozen numerical source changed.')
    cases = read(OUT / ('prepared_configs.recovery.json' if resume else 'prepared_configs.json'))
    for case in cases:
        config = read(ROOT / case['path'])
        if any(config['completion'][k] != v for k, v in completion_counts(config).items()):
            raise ValueError('Declared completion counts disagree with requested coverage.')
    started = time.monotonic()
    previous_seconds = 0.0
    summary = dict(study_id=STUDY, state='running', pid=os.getpid(), target='main:RTX3090',
        started_at=datetime.now(timezone.utc).isoformat(), runs=[], cap_gpu_hours=1,
        training_started=False, official_test_read=False)
    if resume:
        previous = read(OUT / 'execution.at_failure.json')
        if previous['state'] != 'failed' or sha(OUT / 'execution.at_failure.json') != inputs['previous_execution_sha256']:
            raise ValueError('Recovery does not match the recorded failure.')
        previous_seconds = previous['elapsed_seconds']
        for case in previous['runs']:
            if checked(case)['result_sha256'] != case['result_sha256']:
                raise ValueError('Previously completed result changed.')
        summary.update(started_at=previous['started_at'], resumed_at=datetime.now(timezone.utc).isoformat(),
            runs=previous['runs'], initial_smoke=previous['smoke'], recovery=inputs['recovery'],
            previous_charged_gpu_hours=previous['charged_gpu_hours'])
    def run_case(case, *, smoke=False):
        phase = 'smoke' if smoke else 'runs'
        directory = OUT / phase / case['name']
        remaining = 3600 - previous_seconds - (time.monotonic() - started)
        wall_remaining = 3600 - (datetime.now(timezone.utc) - datetime.fromisoformat(summary['started_at'])).total_seconds()
        remaining = min(remaining, wall_remaining)
        if directory.exists() or remaining < 180:
            raise ValueError('Existing output or exhausted budget.')
        command = [sys.executable, str(RUNNER), '--config', str(ROOT / case['path']),
                   '--output-root', str(directory.parent), '--run-id', case['name'],
                   '--device', 'cuda', '--target', 'main:RTX3090']
        if smoke:
            command.append('--smoke')
        summary.update(current_case=case['name'], current_phase=phase,
                       elapsed_seconds=previous_seconds + time.monotonic() - started)
        write(OUT / 'execution.json', summary)
        print('START', phase, case['name'], flush=True)
        with (OUT / f'{phase}_{case["name"]}.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=min(180 if smoke else 600, remaining - 30))
        result = checked(case, smoke=smoke)
        result['native_regression'] = regression(case, smoke=smoke)
        if smoke:
            summary['smoke'] = result
        else:
            summary['runs'].append(result)
        summary['elapsed_seconds'] = previous_seconds + time.monotonic() - started
        write(OUT / 'execution.json', summary)
        print('DONE', phase, case['name'], flush=True)
    try:
        smoke_case = dict(cases[0], name=cases[0]['name'] + '_recovery_smoke') if resume else cases[0]
        run_case(smoke_case, smoke=True)
        completed = {c['name'] for c in summary['runs']}
        for case in cases:
            if case['name'] not in completed:
                run_case(case)
        summary.update(state='complete', current_case=None, current_phase=None,
                       completed_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        summary.update(state='failed', error=repr(exc))
        raise
    finally:
        summary['elapsed_seconds'] = previous_seconds + time.monotonic() - started
        summary['charged_gpu_hours'] = summary['elapsed_seconds'] / 3600
        write(OUT / 'execution.json', summary)
        (OUT / 'gpu_hours.csv').write_text('study,cap_gpu_hours,charged_gpu_hours,state\n'
            f'{STUDY},1,{summary["charged_gpu_hours"]:.12f},{summary["state"]}\n')
        print({k: summary.get(k) for k in ('state', 'elapsed_seconds', 'charged_gpu_hours')}, flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('prepare', 'run', 'resume'):
        raise SystemExit('Usage: python -m experiments.run_phase_displacement_t_sweep prepare|run|resume')
    if sys.argv[1] == 'prepare':
        prepare()
    else:
        run(resume=sys.argv[1] == 'resume')
