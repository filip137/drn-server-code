"""Read-only Conv3 baseline T sweep at fixed beta100/K8, then confirmation."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.reporting import validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'eqprop-baseline-wide-t-relaxation-20260916-v1'
OUT = ROOT / 'results' / STUDY
PRIOR = ROOT / 'results/eqprop-beta-selection-audit-20260914-v1'
OLD_CONFIGS = ROOT / 'configs/conv/eqprop_beta_selection_audit_20260914_v1/v2'
CONFIGS = ROOT / 'configs/conv/eqprop_baseline_wide_t_20260916_v1'
RUNNER = OUT / 'source/experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py'
TS = (12, 16, 24, 32)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, frozen=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if frozen and path.exists():
        if path.read_text() != content:
            raise ValueError(f'Refusing to change frozen input: {path}')
        return
    temporary = path.with_suffix('.tmp')
    temporary.write_text(content)
    temporary.replace(path)


def prepare():
    if STUDY not in (ROOT / 'docs/current_simulations.md').read_text():
        raise ValueError('Register the persistent study row before preparation.')
    frozen = read(PRIOR / 'source_identity.json')
    for name, expected in frozen.items():
        source, dest = PRIOR / 'source' / name, OUT / 'source' / name
        if sha(source) != expected:
            raise ValueError(f'Preserved analyzer changed: {name}')
        if dest.exists() and sha(dest) != expected:
            raise ValueError(f'Existing snapshot differs: {dest}')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    write(OUT / 'source_identity.json', frozen, frozen=True)
    template_path = OLD_CONFIGS / 'selection_conv3_baseline_seed0_beta100.json'
    original_argv = sys.argv[:]
    sys.argv = [str(RUNNER), '--config', str(template_path)]
    try:
        spec = importlib.util.spec_from_file_location('_baseline_t_gate', RUNNER)
        gate = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = gate
        spec.loader.exec_module(gate)
    finally:
        sys.argv = original_argv
    base = gate.base
    template = read(template_path)
    source_config = read(Path(template['cases'][0]['run_dir']) / 'config.used.json')
    signature = base._dataset_signature(source_config,
        data_root=Path(template['dataset']['root']), batch_size=16)
    loader = base._resolve_callable(signature['factory'])(
        **dict(signature['params'], device=base.torch.device('cpu'))).build()
    old = read(PRIOR / 'cohorts/partition.json')
    seen = set(old['historical'] + old['selection'] + old['confirmation'])
    pool = [int(i) for i in loader.validation_indices if int(i) not in seen]
    rng = base.np.random.Generator(base.np.random.PCG64(2026091601))
    fresh = [int(i) for i in rng.permutation(pool)[:512]]
    assert len(fresh) == len(set(fresh)) == 512 and not seen.intersection(fresh)
    indices = old['historical'] + fresh
    excluded = old['selection'] + old['confirmation']
    batches, cohort = base._build_validation_cohort(source_config,
        data_root=Path(template['dataset']['root']), batch_size=16,
        example_count=576, source_indices=indices, excluded_source_indices=excluded)
    guards = [{k: b[k] for k in ('batch_index', 'source_indices_sha256', 'payload_sha256')}
              for b in batches]
    if guards[:4] != template['dataset']['expected_batch_guards'][:4]:
        raise ValueError('The historical input batches changed.')
    confirmation = dict(template['dataset'], source_indices=indices,
        excluded_source_indices=excluded, expected_batch_guards=guards,
        cohort_role='fresh_confirmation_20260916')
    write(OUT / 'cohorts/confirmation.json', cohort, frozen=True)
    write(OUT / 'cohorts/partition.json', dict(seed=2026091601,
        algorithm='NumPy PCG64', historical=old['historical'], fresh=fresh,
        excluded_prior_audit=sorted(seen), official_test_read=False,
        validation_indices_sha256=loader.validation_indices_hash), frozen=True)
    prepared = []
    for phase in ('selection', 'confirmation'):
        for seed in ([0] if phase == 'selection' else [0, 1, 2]):
            template_name = f'{phase}_conv3_baseline_seed{seed}_beta100.json'
            for t in TS:
                config = copy.deepcopy(read(OLD_CONFIGS / template_name))
                config.update(study_id=STUDY,
                    scientific_question='Does increasing free-phase T alone fix wide Conv3 baseline beta100/K8 gradient and equilibrium qualification?')
                config['cases'][0].update(replay_T=t, replay_K=8)
                if phase == 'confirmation':
                    config['dataset'] = confirmation
                name = f'{phase}_conv3_baseline_seed{seed}_T{t}_K8_beta100'
                path = CONFIGS / f'{name}.json'
                write(path, config, frozen=True)
                prepared.append(dict(name=name, phase=phase, seed=seed, T=t, K=8,
                    beta=100, path=str(path.relative_to(ROOT)), sha256=sha(path)))
    write(OUT / 'prepared_configs.json', prepared, frozen=True)
    reference = PRIOR / 'selection/selection_conv3_baseline_seed0_beta100'
    if errors := validate_run(reference):
        raise ValueError(errors)
    write(OUT / 'reference.json', dict(result=str((reference / 'result.json').relative_to(ROOT)),
        result_sha256=sha(reference / 'result.json'), T=8, K=8, beta=100,
        config_sha256=sha(template_path)), frozen=True)
    write(OUT / 'execution_inputs.json', dict(driver_sha256=sha(__file__),
        plan_sha256=sha(ROOT / 'docs/eqprop_baseline_read_noise_beta_plan_20260916.md'),
        cap_gpu_hours=2, selection_cases=4, confirmation_cases_maximum=3), frozen=True)
    print(json.dumps(dict(state='prepared', configs=len(prepared), fresh_examples=len(fresh),
                         validation_indices_sha256=loader.validation_indices_hash)))


def checked(case, *, smoke=False):
    if sha(ROOT / case['path']) != case['sha256']:
        raise ValueError('Frozen config changed.')
    directory = OUT / ('smoke' if smoke else case['phase']) / case['name']
    if errors := validate_run(directory):
        raise ValueError(f'{directory}: {errors}')
    manifest = read(directory / 'manifest.json')
    if manifest['configuration']['sha256'] != case['sha256']:
        raise ValueError('Result/config identity mismatch.')
    result = read(directory / 'result.json')
    metrics = result['terminal_metrics']
    count = 1 if smoke else 72
    if metrics['replay_count'] != count or metrics['layer_comparison_count'] != 4 * count:
        raise ValueError('Incomplete replay coverage.')
    if (metrics['official_test_read'] or metrics['optimizer_steps_applied']
            or not metrics['source_bytes_unchanged'] or not metrics['all_bias_tensors_exact_zero']
            or metrics['endpoint_read_noise_std'] != 0):
        raise ValueError('Read-only clean replay guards failed.')
    return dict(case, result=str((directory / 'result.json').relative_to(ROOT)),
                result_sha256=sha(directory / 'result.json'), **metrics)


def run():
    if (OUT / 'execution.json').exists():
        raise ValueError('Execution exists; reconcile before any explicit recovery.')
    inputs = read(OUT / 'execution_inputs.json')
    if inputs['driver_sha256'] != sha(__file__):
        raise ValueError('Prepared driver changed.')
    for name, expected in read(OUT / 'source_identity.json').items():
        if sha(OUT / 'source' / name) != expected:
            raise ValueError('Frozen analyzer changed.')
    cases = read(OUT / 'prepared_configs.json')
    started = time.monotonic()
    deadline = started + 7200
    summary = dict(study_id=STUDY, state='running', pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(), target='main:RTX3090',
        cap_gpu_hours=2, runs=[], selected_T=None, official_test_read=False,
        optimizer_steps_applied=False, training_started=False)

    def run_case(case, *, smoke=False):
        phase = 'smoke' if smoke else case['phase']
        directory = OUT / phase / case['name']
        if directory.exists():
            raise ValueError(f'Existing output needs explicit recovery: {directory}')
        remaining = deadline - time.monotonic()
        if remaining < 180:
            raise TimeoutError('Study budget exhausted before next case.')
        command = [sys.executable, str(RUNNER), '--config', str(ROOT / case['path']),
            '--output-root', str(directory.parent), '--run-id', case['name'],
            '--device', 'cuda', '--target', 'main:RTX3090']
        if smoke:
            command.append('--smoke')
        summary.update(current_case=case['name'], current_phase=phase,
                       elapsed_seconds=time.monotonic() - started)
        write(OUT / 'execution.json', summary)
        print(f'START {phase} {case["name"]}', flush=True)
        with (OUT / f'{phase}_{case["name"]}.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                check=True, timeout=min(180 if smoke else 900, remaining - 30))
        result = checked(case, smoke=smoke)
        if smoke:
            summary['smoke'] = result
        else:
            summary['runs'].append(result)
        summary['elapsed_seconds'] = time.monotonic() - started
        write(OUT / 'execution.json', summary)
        print(f'DONE {phase} T={case["T"]} seed={case["seed"]}: '
            f'cosine={result["minimum_cosine"]:.6f} '
            f'gradient={result["gradient_fidelity_all_passed"]} '
            f'equilibrium={result["equilibrium_residual_all_passed"]}', flush=True)
        return result

    try:
        run_case(cases[0], smoke=True)
        for case in cases:
            if case['phase'] == 'selection':
                run_case(case)
        passing = [r for r in summary['runs'] if r['unqualified_launch_gate_all_passed']]
        selected = min((r['T'] for r in passing), default=None)
        decision = dict(selected_T=selected, beta=100, K=8,
            frozen_before_confirmation=True,
            state='confirmation_pending' if selected else 'no_passing_T',
            selection_results=[dict(result=r['result'], sha256=r['result_sha256'])
                               for r in summary['runs']])
        write(OUT / 'selection_decision.json', decision, frozen=True)
        summary.update(selected_T=selected, decision=decision['state'])
        write(OUT / 'execution.json', summary)
        if selected is not None:
            confirmations = [run_case(c) for c in cases
                             if c['phase'] == 'confirmation' and c['T'] == selected]
            assert len(confirmations) == 3
            summary['decision'] = ('confirmed' if all(r['unqualified_launch_gate_all_passed']
                for r in confirmations) else 'confirmation_failed')
        summary.update(state='complete', current_case=None, current_phase=None,
                       completed_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        summary.update(state='failed', error=repr(exc))
        raise
    finally:
        summary['elapsed_seconds'] = time.monotonic() - started
        summary['charged_gpu_hours'] = summary['elapsed_seconds'] / 3600
        write(OUT / 'execution.json', summary)
        (OUT / 'gpu_hours.csv').write_text('study,cap_gpu_hours,charged_gpu_hours,state\n'
            f'{STUDY},2,{summary["charged_gpu_hours"]:.12f},{summary["state"]}\n')
        print(json.dumps({k: summary.get(k) for k in
            ('state', 'decision', 'selected_T', 'elapsed_seconds', 'charged_gpu_hours')}), flush=True)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('prepare', 'run'):
        raise SystemExit('Usage: python -m experiments.run_baseline_wide_t_audit prepare|run')
    (prepare if sys.argv[1] == 'prepare' else run)()
