"""Read-only phase displacement for Conv1/2/3 baseline, ours and legacy."""
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
STUDY = 'eqprop-phase-displacement-conv123-20260916-v1'
OUT = ROOT / 'results' / STUDY
BETA = ROOT / 'results/eqprop-conv3-baseline-beta-tk8-20260916-v1'
PRIOR = ROOT / 'results/eqprop-beta-selection-audit-20260914-v1'
PARENT = ROOT / 'results/paper-training-completion-20260911-v1'
CONFIGS = ROOT / 'configs/conv/eqprop_phase_displacement_20260916_v1'
PLAN = ROOT / 'docs/eqprop_phase_displacement_plan_20260916.md'
RUNNER = OUT / 'source/experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py'


def prepare():
    if STUDY not in (ROOT / 'docs/current_simulations.md').read_text():
        raise ValueError('Register the persistent study row first.')
    beta_result = read(BETA / 'execution.json')
    if beta_result['state'] != 'complete' or beta_result['selected_beta'] is None:
        raise ValueError('Finish the beta sweep and record a candidate before this study.')
    frozen = read(PRIOR / 'source_identity.json')
    snapshot = {}
    for name, expected in frozen.items():
        prior = PRIOR / 'source' / name
        if sha(prior) != expected:
            raise ValueError('Preserved analyzer changed.')
        source = ROOT / name if name.endswith('analyze_conv_eqprop_bptt_beta_tk_displacement.py') else prior
        dest = OUT / 'source' / name
        if dest.exists() and sha(dest) != sha(source):
            raise ValueError('An existing snapshot differs.')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        snapshot[name] = sha(dest)
    write(OUT / 'source_identity.json', snapshot, frozen=True)
    betas = {'conv1': {'baseline':100, 'ours':30, 'legacy':3},
             'conv2': {'baseline':100, 'ours':10, 'legacy':.03},
             'conv3': {'baseline':beta_result['selected_beta'], 'ours':3, 'legacy':.001}}
    prepared = []
    for architecture in ('conv3', 'conv2', 'conv1'):
        depth = int(architecture[-1])
        template = read(ROOT / f'configs/conv/eqprop_beta_selection_audit_20260914_v1/v2/selection_{architecture}_baseline_seed0_beta100.json')
        for scheme in ('baseline', 'ours', 'legacy'):
            config = copy.deepcopy(template)
            source = ROOT / f'paper_ready_results/bundles/table1_wide_bptt/{architecture}/{scheme}/seed0'
            manifest = read(source / 'manifest.json')
            asset = PARENT / f'assets/wide_kaiming/{architecture}/seed0.pt'
            beta = betas[architecture][scheme]
            scale = {'baseline':1, 'ours':4, 'legacy':16}[scheme] ** depth
            config.update(study_id=STUDY, scientific_question='How large is free-to-nudged per-layer voltage displacement across Conv1/2/3 amplification schemes under the declared beta contracts?')
            config['source_contract']['source_study_root'] = str(source.parent)
            config['cases'] = [dict(architecture=architecture, scheme=scheme, model_seed=0,
                run_id='seed0', run_dir=str(source), T={1:4,2:6,3:8}[depth], K={1:4,2:6,3:8}[depth],
                output_row_exponent_L=depth, base_beta=beta/scale, injected_beta=beta,
                production_source_commit=manifest['git']['commit'],
                production_source_archive_sha256=manifest['git'].get('source_archive_sha256'),
                source_file_sha256={n:sha(source/n) for n in config['source_contract']['required_source_files']},
                initializer_checkpoint_path=str(asset), initializer_checkpoint_sha256=sha(asset))]
            name = f'{architecture}_{scheme}_seed0'
            path = CONFIGS / f'{name}.json'
            write(path, config, frozen=True)
            prepared.append(dict(name=name, architecture=architecture, scheme=scheme, seed=0,
                beta=beta, base_beta=beta/scale, T=config['cases'][0]['T'], K=config['cases'][0]['K'],
                path=str(path.relative_to(ROOT)), sha256=sha(path)))
    write(OUT / 'prepared_configs.json', prepared, frozen=True)
    shutil.copy2(PLAN, OUT / 'plan.frozen.md')
    write(OUT / 'execution_inputs.json', dict(driver_sha256=sha(__file__),
        helper_sha256=sha(ROOT / 'experiments/run_baseline_wide_t_audit.py'),
        plan_sha256=sha(OUT/'plan.frozen.md'), beta_result_sha256=sha(BETA/'execution.json'),
        baseline_conv3_beta_decision=beta_result['decision'], cap_gpu_hours=1, cases=9), frozen=True)
    print('Prepared nine cases; Conv3 baseline beta',beta_result['selected_beta'],beta_result['decision'])


def checked(case, *, smoke=False):
    directory = OUT / ('smoke' if smoke else 'runs') / case['name']
    if sha(ROOT / case['path']) != case['sha256']:
        raise ValueError('Frozen config changed.')
    if errors := validate_run(directory):
        raise ValueError(f'{directory}: {errors}')
    manifest = read(directory/'manifest.json')
    if manifest['configuration']['sha256'] != case['sha256']:
        raise ValueError('Result/config mismatch.')
    m = read(directory/'result.json')['terminal_metrics']
    replays = 1 if smoke else 72
    depth = int(case['architecture'][-1])
    if m['replay_count'] != replays or m['layer_comparison_count'] != replays*(depth+1):
        raise ValueError('Incomplete replay coverage.')
    if (m['official_test_read'] or m['optimizer_steps_applied'] or m['endpoint_read_noise_std'] != 0
            or not m['source_bytes_unchanged'] or not m['all_bias_tensors_exact_zero']):
        raise ValueError('Clean read-only guards failed.')
    rows = list(csv.DictReader((directory/'state_displacement.csv').open()))
    if len(rows) != replays*(depth+2)*5:
        raise ValueError('Missing phase/reference displacement rows.')
    for row in rows:
        if row['state_layer_name'] != '__all__' and (not row['current_sum'] or not row['reference_sum']):
            raise ValueError('Missing signed voltage statistics.')
    return dict(case,result=str((directory/'result.json').relative_to(ROOT)),
                result_sha256=sha(directory/'result.json'),**m)


def smoke_regression(case):
    # Both smokes use the same first trained baseline Conv3 batch and beta.
    old_smoke = read(BETA/'execution.json')['smoke']
    if case['beta'] != old_smoke['beta']:
        raise ValueError('Need an explicit same-beta regression reference before execution.')
    old = (ROOT/old_smoke['result']).parent
    new = OUT/'smoke'/case['name']
    for name in ('layer_metrics.csv','state_displacement.csv','equilibrium_residuals.csv'):
        before = list(csv.DictReader((old/name).open()))
        after = list(csv.DictReader((new/name).open()))
        if len(before) != len(after) or any(any(a[k] != b[k] for k in a) for a,b in zip(before,after,strict=True)):
            raise ValueError(f'Added statistics changed shared smoke measurements: {name}')
    write(OUT/'smoke_regression.json',dict(shared_measurements_identical=True,
        reference_result=old_smoke['result'],reference_sha256=sha(old/'result.json'),
        new_result=str((new/'result.json').relative_to(ROOT)),new_sha256=sha(new/'result.json')))


def run():
    if (OUT/'execution.json').exists():
        raise ValueError('Existing execution needs explicit reconciliation.')
    inputs=read(OUT/'execution_inputs.json')
    if inputs['driver_sha256']!=sha(__file__) or inputs['helper_sha256']!=sha(ROOT/'experiments/run_baseline_wide_t_audit.py'):
        raise ValueError('Frozen driver/helper changed.')
    for name,expected in read(OUT/'source_identity.json').items():
        if sha(OUT/'source'/name)!=expected: raise ValueError('Frozen analyzer changed.')
    cases=read(OUT/'prepared_configs.json'); started=time.monotonic()
    summary=dict(study_id=STUDY,state='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
        target='main:RTX3090',runs=[],cap_gpu_hours=1,official_test_read=False,training_started=False)
    def run_case(case,smoke=False):
        phase='smoke' if smoke else 'runs'; directory=OUT/phase/case['name']
        remaining=3600-(time.monotonic()-started)
        if directory.exists() or remaining<180: raise ValueError('Existing output or exhausted budget.')
        command=[sys.executable,str(RUNNER),'--config',str(ROOT/case['path']),
            '--output-root',str(directory.parent),'--run-id',case['name'],'--device','cuda','--target','main:RTX3090']
        if smoke: command.append('--smoke')
        summary.update(current_case=case['name'],current_phase=phase,elapsed_seconds=time.monotonic()-started)
        write(OUT/'execution.json',summary); print('START',phase,case['name'],flush=True)
        with (OUT/f'{phase}_{case["name"]}.log').open('x') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
                           timeout=min(180 if smoke else 600,remaining-30))
        result=checked(case,smoke=smoke)
        if smoke: summary['smoke']=result
        else: summary['runs'].append(result)
        summary['elapsed_seconds']=time.monotonic()-started
        write(OUT/'execution.json',summary); print('DONE',phase,case['name'],flush=True)
    try:
        run_case(cases[0],smoke=True); smoke_regression(cases[0])
        for case in cases: run_case(case)
        summary.update(state='complete',current_case=None,current_phase=None,completed_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        summary.update(state='failed',error=repr(exc)); raise
    finally:
        summary['elapsed_seconds']=time.monotonic()-started
        summary['charged_gpu_hours']=summary['elapsed_seconds']/3600
        write(OUT/'execution.json',summary)
        (OUT/'gpu_hours.csv').write_text('study,cap_gpu_hours,charged_gpu_hours,state\n'
            f'{STUDY},1,{summary["charged_gpu_hours"]:.12f},{summary["state"]}\n')
        print({k:summary.get(k) for k in ('state','elapsed_seconds','charged_gpu_hours')},flush=True)


if __name__=='__main__':
    if len(sys.argv)!=2 or sys.argv[1] not in ('prepare','run'):
        raise SystemExit('Usage: python -m experiments.run_phase_displacement prepare|run')
    (prepare if sys.argv[1]=='prepare' else run)()
