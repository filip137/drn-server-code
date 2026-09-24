"""Prepare, run and analyze the fixed seed-0 beta selection-rule comparison."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.reporting import atomic_write_json, sha256_file, utc_now, validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'eqprop-beta-rule-comparison-20260918-v1'
OUT = ROOT / 'results' / STUDY
CONFIGS = ROOT / 'configs/conv/eqprop_beta_rule_comparison_20260918_v1'
PRIOR = ROOT / 'results/eqprop-beta-selection-audit-20260914-v1'
PARENT = ROOT / 'results/paper-training-completion-20260911-v1'
PLAN = ROOT / 'docs/eqprop_beta_rule_comparison_plan_20260918.md'
RUNNER_NAME = 'analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py'
RUNNER = OUT / 'source/experiments' / RUNNER_NAME
FACTORS = (.01, .03, .1, .3, 1, 2, 3, 5, 7.5, 10, 20, 30, 50, 75, 100, 300, 1000)
ANCHORS = {'conv1': {'baseline': 100, 'ours': 30, 'legacy': 3},
           'conv2': {'baseline': 100, 'ours': 10, 'legacy': .03},
           'conv3': {'baseline': 10, 'ours': 3, 'legacy': .001}}
ROLES = ('reconstructed_initialization', 'best_validation')
CAP_SECONDS = 6 * 3600


def read(path):
    return json.loads(Path(path).read_text())


def frozen_json(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f'Frozen file changed: {path}')
    atomic_write_json(path, value)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def prepare():
    if STUDY not in (ROOT / 'docs/current_simulations.md').read_text():
        raise ValueError('Register the planned study before creating its directory.')
    if (OUT / 'execution.json').exists():
        raise ValueError('Execution already exists; do not replace its inputs.')
    source_names = [RUNNER_NAME, 'audit_eqprop_float64_shadow.py',
                    'analyze_conv_eqprop_bptt_beta_tk_displacement.py',
                    'analyze_conv_eqprop_bptt_checkpoint_gradients.py']
    identities = {}
    for name in source_names:
        source = ROOT / 'experiments' / name
        destination = OUT / 'source/experiments' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and sha256_file(source) != sha256_file(destination):
            raise ValueError('An existing source snapshot differs.')
        shutil.copy2(source, destination)
        identities[f'experiments/{name}'] = sha256_file(destination)
    frozen_json(OUT / 'source_identity.json', identities)
    cases = []
    for arch in ('conv3', 'conv2', 'conv1'):
        template = read(ROOT / 'configs/conv/eqprop_beta_selection_audit_20260914_v1/v2' /
                        f'selection_{arch}_baseline_seed0_beta100.json')
        template['source_contract']['runtime_source_root'] = str(PARENT / 'source')
        for name, expected in template['source_contract']['runtime_code_sha256'].items():
            if sha256_file(PARENT / 'source' / name) != expected:
                raise ValueError(f'Preserved numerical runtime changed: {name}')
        for scheme, anchor in ANCHORS[arch].items():
            source = ROOT / f'paper_ready_results/bundles/table1_wide_bptt/{arch}/{scheme}/seed0'
            if errors := validate_run(source):
                raise ValueError(f'{source}: {errors}')
            manifest = read(source / 'manifest.json')
            depth = int(arch[-1])
            scale = {'baseline': 1, 'ours': 4, 'legacy': 16}[scheme] ** depth
            for factor in FACTORS:
                beta = float(format(anchor * factor, '.12g'))
                config = copy.deepcopy(template)
                config.update(study_id=STUDY, record_whole_gradient=True,
                    scientific_question='Compare clean beta limits for layerwise versus concatenated gradient cosine .90/.95/.99, with and without the norm gate.')
                case = config['cases'][0]
                case.update(scheme=scheme, run_dir=str(source), base_beta=beta / scale,
                    injected_beta=beta,
                    production_source_commit=manifest['git']['commit'],
                    production_source_archive_sha256=manifest['git'].get('source_archive_sha256'),
                    source_file_sha256={n: sha256_file(source / n)
                                       for n in config['source_contract']['required_source_files']})
                name = f'{arch}_{scheme}_beta{beta:g}'.replace('.', 'p')
                path = CONFIGS / f'{name}.json'
                frozen_json(path, config)
                cases.append(dict(name=name, architecture=arch, scheme=scheme,
                                  anchor=anchor, factor=factor, beta=beta,
                                  config=str(path.relative_to(ROOT)), sha256=sha256_file(path)))
    assert len(cases) == 153 and len({c['name'] for c in cases}) == 153
    frozen_json(OUT / 'cases.json', cases)
    destination = OUT / 'plan.frozen.md'
    if destination.exists() and sha256_file(destination) != sha256_file(PLAN):
        raise ValueError('Frozen plan changed.')
    shutil.copy2(PLAN, destination)
    frozen_json(OUT / 'inputs.json', dict(study_id=STUDY, case_count=153,
        replay_count=11016, layer_comparison_count=33048, cap_gpu_hours=6,
        plan_sha256=sha256_file(PLAN), driver_sha256=sha256_file(__file__),
        source_identity_sha256=sha256_file(OUT / 'source_identity.json'),
        cohort_sha256=sha256_file(PRIOR / 'cohorts/selection.json'),
        official_test_read=False, optimizer_steps_applied=False))
    print('Prepared 153 immutable configurations; 11,016 replays; seed 0 only.')


def verify_inputs():
    inputs = read(OUT / 'inputs.json')
    if sha256_file(__file__) != inputs['driver_sha256']:
        raise ValueError('Prepared driver changed.')
    for name, expected in read(OUT / 'source_identity.json').items():
        if sha256_file(OUT / 'source' / name) != expected:
            raise ValueError('Frozen analyzer changed.')
    for case in read(OUT / 'cases.json'):
        if sha256_file(ROOT / case['config']) != case['sha256']:
            raise ValueError('Frozen config changed.')


def checked(case, directory, smoke=False):
    if errors := validate_run(directory):
        raise ValueError(f'{directory}: {errors}')
    manifest = read(directory / 'manifest.json')
    if manifest['configuration']['sha256'] != case['sha256']:
        raise ValueError('Result/config mismatch.')
    metrics = read(directory / 'result.json')['terminal_metrics']
    expected = 1 if smoke else 72
    if (metrics['replay_count'] != expected or
        metrics['layer_comparison_count'] != expected * (int(case['architecture'][-1]) + 1)):
        raise ValueError('Incomplete replay coverage.')
    if (metrics['official_test_read'] or metrics['optimizer_steps_applied'] or
        not metrics['source_bytes_unchanged'] or not metrics['all_bias_tensors_exact_zero'] or
        metrics['endpoint_read_noise_std'] != 0):
        raise ValueError('Clean read-only guards failed.')
    whole = csv_rows(directory / 'whole_gradient_metrics.csv')
    if len(whole) != expected:
        raise ValueError('Incomplete whole-gradient coverage.')
    return dict(case, state='complete', directory=str(directory.relative_to(ROOT)),
                result_sha256=sha256_file(directory / 'result.json'), **metrics)


def run_case(case, directory, timeout, smoke=False):
    if directory.exists():
        raise ValueError(f'Refusing to replace existing output: {directory}')
    command = [sys.executable, str(RUNNER), '--config', str(ROOT / case['config']),
               '--output-root', str(directory.parent), '--run-id', directory.name,
               '--device', 'cuda', '--target', 'local:RTX3090']
    if smoke:
        command.append('--smoke')
    log_path = OUT / 'logs' / f'{directory.parent.name}_{directory.name}.log'
    log_path.parent.mkdir(exist_ok=True)
    started = time.monotonic()
    with log_path.open('x') as log:
        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=timeout)
    elapsed = time.monotonic() - started
    if process.returncode:
        # A diagnosed nonfinite numerical outcome is retained as a rejected beta.
        # Infrastructure, provenance, timeout and implementation failures stop the driver.
        status_path = directory / 'status.json'
        message = log_path.read_text()
        if (not smoke and status_path.is_file() and read(status_path)['state'] == 'failed'
                and any(s in message for s in ('NonFiniteTrainingError', 'contains non-finite',
                                              'Non-finite', 'non-finite values'))
                and 'out of memory' not in message.lower()):
            return dict(case, state='numerical_failure', directory=str(directory.relative_to(ROOT)),
                        elapsed_seconds=elapsed, log=str(log_path.relative_to(ROOT)))
        raise RuntimeError(f'Case failed ({process.returncode}): {log_path}')
    return dict(checked(case, directory, smoke), elapsed_seconds=elapsed)


def smoke():
    verify_inputs()
    case = next(c for c in read(OUT / 'cases.json') if c['architecture'] == 'conv3'
                and c['scheme'] == 'baseline' and c['factor'] == 1)
    result = run_case(case, OUT / 'smoke' / case['name'], timeout=180, smoke=True)
    atomic_write_json(OUT / 'smoke.json', result)
    print(json.dumps(result, indent=2))


def execute():
    verify_inputs()
    if (OUT / 'execution.json').exists():
        raise ValueError('Existing execution requires explicit recovery reconciliation.')
    smoke_result = read(OUT / 'smoke.json')
    checked(smoke_result, ROOT / smoke_result['directory'], smoke=True)
    started = time.monotonic()
    summary = dict(study_id=STUDY, state='running', pid=os.getpid(), target='local:RTX3090',
        started_at=utc_now(), expected_cases=153, runs=[], smoke_seconds=smoke_result['elapsed_seconds'],
        cap_gpu_hours=6, official_test_read=False, training_started=False)
    atomic_write_json(OUT / 'execution.json', summary)
    try:
        # Historical overlap and smaller betas first; all predeclared points are measured.
        cases = read(OUT / 'cases.json')
        for case in cases:
            remaining = CAP_SECONDS - smoke_result['elapsed_seconds'] - (time.monotonic() - started)
            if remaining < 180:
                raise TimeoutError('Physical GPU-hour cap reached before the next case.')
            summary.update(current_case=case['name'], updated_at=utc_now())
            atomic_write_json(OUT / 'execution.json', summary)
            print('START ' + case['name'], flush=True)
            result = run_case(case, OUT / 'production' / case['name'], min(900, remaining - 30))
            summary['runs'].append(result)
            summary.update(elapsed_seconds=time.monotonic() - started, updated_at=utc_now())
            atomic_write_json(OUT / 'execution.json', summary)
            print(f'DONE {len(summary["runs"])}/153 {case["name"]} '
                  f'{result["state"]} {result["elapsed_seconds"]:.1f}s', flush=True)
        summary.update(state='complete', current_case=None, completed_at=utc_now())
    except BaseException as error:
        summary.update(state='failed', error=repr(error))
        raise
    finally:
        summary['elapsed_seconds'] = time.monotonic() - started
        summary['charged_gpu_hours'] = (summary['elapsed_seconds'] + summary['smoke_seconds']) / 3600
        atomic_write_json(OUT / 'execution.json', summary)


def passes(rows, threshold, norm_gate):
    if not rows:
        return False
    for row in rows:
        try:
            cosine = float(row['cosine'])
            delta = float(row['symmetric_norm_delta'])
        except (ValueError, TypeError):
            return False
        if not math.isfinite(cosine) or not math.isfinite(delta):
            return False
        if cosine < threshold or (norm_gate and delta > .10):
            return False
    return True


def select_beta(points):
    """Keep every tested point; never infer monotonicity or an exact boundary."""
    points = sorted(points)
    passing = [beta for beta, passed in points if passed]
    regions = []
    for beta, passed in points:
        if passed:
            if not regions or regions[-1]['closed']:
                regions.append(dict(low=beta, high=beta, closed=False))
            regions[-1]['high'] = beta
        elif regions:
            regions[-1]['closed'] = True
    return dict(selected_beta=max(passing) if passing else None,
                upper_edge_open=bool(points and points[-1][1]),
                passing_regions=[dict(low=r['low'], high=r['high']) for r in regions],
                passing_count=len(passing))


def analyze():
    # Analysis lives in a separate module so plotting cannot change frozen execution inputs.
    from experiments.analyze_beta_rule_comparison import main
    main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'smoke', 'run', 'analyze'))
    args = parser.parse_args()
    {'prepare': prepare, 'smoke': smoke, 'run': execute, 'analyze': analyze}[args.action]()
