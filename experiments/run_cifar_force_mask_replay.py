"""Stage, collect and analyze exp-030: float32 replays with the inactive-channel force mask.

Each exp-025 checkpoint is replayed twice in float32: ``masked`` compares all
blocks with ``force_mask='inactive_output_channels'``; ``control`` compares
block3 only without the mask. Inputs, cohort, targets and calibration are
exp-025's. The runner is ``experiments.compare_cifar_block_beta_gradients``.

    python -m experiments.run_cifar_force_mask_replay stage
    python -m experiments.run_cifar_force_mask_replay collect CASE ARM   # queue completion
    python -m experiments.run_cifar_force_mask_replay analyze
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
STUDY = 'cifar-force-mask-float32-replay-20261001-v1'
ROOT = REPO / 'results' / STUDY
PARENT = REPO / 'results/cifar-layer-rms-all-schemes-float64-20260930-v1'
PARENT_CONFIGS = REPO / 'configs/cifar/layer_rms_all_schemes_float64_20260930'
CONFIGS = Path('configs/cifar/force_mask_float32_replay_20261001')
HANDOFF = REPO / 'campaigns/cifar10-beta-selection/series/001-beta-selection/experiments/exp-030-float32-force-mask-replay.md'
ARMS = ('masked', 'control')
REMOTE_RESULTS = Path('/home/filip/server_code/results')
LOCAL_PYTHON = PARENT / 'runtime/py312/bin/python'
REMOTE_PYTHON = Path('/home/filip/miniconda3/envs/py312/bin/python')
PLACEMENTS = ('loulou', 'trex', 'local')
RUNNER_SECONDS = {'masked': 1500, 'control': 900}
QUEUE_SECONDS = {'masked': 1800, 'control': 1200}
EXPECTED_ROWS = {'masked': (168, 1344), 'control': (42, 336)}
OURS_TRAINING_CASES = ('init-ours-e0', 'eqprop-ours-e1', 'eqprop-ours-e5')
ENV = dict(KMP_DISABLE_SHM='1', KMP_SHM_DISABLE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1',
           CUBLAS_WORKSPACE_CONFIG=':4096:8')


def cases():
    return json.loads((PARENT / 'cases.json').read_text())


def config_name(case, arm):
    return f"{case['id']}-{arm}.json"


def job_id(case, arm):
    return f"cifar-force-mask-fp32-{case['id']}-{arm}-20261001"


def make_config(case, arm):
    config = json.loads((PARENT_CONFIGS / f"{case['id']}.json").read_text())
    assert config['precision'] == 'float64' and config['case_id'] == case['id']
    config.update(study_id=STUDY, precision='float32', arm=arm, budget_seconds=RUNNER_SECONDS[arm])
    if arm == 'masked':
        config['force_mask'] = 'inactive_output_channels'
    else:
        config['diagnostic_blocks'] = [3]
    return config


def source_files():
    for directory in ('experiments', 'labs', 'model', 'training'):
        for path in sorted((REPO / directory).rglob('*.py')):
            if '__pycache__' not in path.parts and 'tests' not in path.relative_to(REPO).parts:
                yield path.relative_to(REPO)


def stage():
    from experiments import reporting
    assert not (ROOT / 'source').exists(), 'Frozen source already staged'
    (REPO / CONFIGS).mkdir(parents=True, exist_ok=True)
    for case in cases():
        for arm in ARMS:
            path = REPO / CONFIGS / config_name(case, arm)
            assert not path.exists(), path
            reporting.atomic_write_json(path, make_config(case, arm))
    source = ROOT / 'source'
    files = [*source_files(), *sorted(p.relative_to(REPO) for p in (REPO / CONFIGS).glob('*.json'))]
    for relative in files:
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, source / relative)
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    identity = dict(commit=commit, dirty=True, study_id=STUDY,
        note=('exp-030 float32 replay of the 18 exp-025 checkpoints. Adds the explicit inactive-channel '
              'force mask; references always use the unmasked force. Uncommitted files are identified by hash.'),
        parent_source_identity_sha256=reporting.sha256_file(PARENT / 'source/source_identity.json'),
        source_sha256={str(r): reporting.sha256_file(source / r) for r in files})
    reporting.atomic_write_json(source / 'source_identity.json', identity)
    rows = []
    for case in cases():
        for arm in ARMS:
            rows.append(dict(id=f"{case['id']}-{arm}", case=case['id'], arm=arm, scheme=case['scheme'],
                             epoch=case['epoch'], trajectory=case['trajectory'], groups=case['groups'],
                             parent_input_root=str(PARENT.relative_to(REPO) / case['input_root']),
                             checkpoint_sha256=case['checkpoint_sha256'], config=str(CONFIGS / config_name(case, arm)),
                             job_id=job_id(case, arm)))
    reporting.atomic_write_json(ROOT / 'cases.json', rows)
    for row in rows:
        reporting.atomic_write_json(ROOT / 'launch' / f"{row['id']}.json", queue_spec(row))
    print(json.dumps(dict(staged=len(rows), source_files=len(files)), indent=1))


def placement(row, target):
    from experiments import reporting
    local = target == 'local'
    base = ROOT if local else REMOTE_RESULTS / STUDY
    parent = PARENT if local else REMOTE_RESULTS / PARENT.name
    python = LOCAL_PYTHON if local else REMOTE_PYTHON
    dataset = (REPO if local else REMOTE_RESULTS.parent) / 'results/cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1/data/cifar-10-batches-py'
    input_root = parent / 'cases' / row['case']
    run_dir = base / 'cells' / row['id']
    config = base / 'source' / row['config']
    identity = base / 'source/source_identity.json'
    return dict(workdir=str(base / 'source'), run_dir=str(run_dir), env=dict(ENV),
        gpu_models=['RTX 3090'] if local else ['RTX 5090'],
        argv=[str(python), '-u', '-m', 'experiments.compare_cifar_block_beta_gradients', '--config', str(config),
              '--input-root', str(input_root), '--output-dir', str(run_dir), '--dataset-root', str(dataset),
              '--source-identity', str(identity), '--device', 'cuda'],
        validate_argv=[str(python), '-m', 'experiments.reporting', 'validate-run', str(run_dir)],
        prepared_files=[dict(path=str(identity), sha256=reporting.sha256_file(ROOT / 'source/source_identity.json')),
                        dict(path=str(config), sha256=reporting.sha256_file(ROOT / 'source' / row['config'])),
                        dict(path=str(input_root / 'inputs/checkpoint.pt'), sha256=row['checkpoint_sha256'])])


def queue_spec(row):
    priority = 3 if row['case'] in OURS_TRAINING_CASES and row['arm'] == 'masked' else (2 if row['arm'] == 'masked' else 1)
    return dict(id=row['job_id'], project_dir=str(REPO), handoff=str(HANDOFF),
        authorization=('Filip asked to implement the inactive-channel force mask, replay all 18 exp-025 checkpoints '
                       'with it and then run a short float32 training comparison. Read-only float32 replays: '
                       'masked all blocks plus unmasked block3 control; no training, optimizer, noise or test split.'),
        resource_profile='default', peak_memory_mib=11264, headroom_mib=2048, priority=priority,
        expected_seconds=900 if row['arm'] == 'masked' else 450, budget_seconds=QUEUE_SECONDS[row['arm']],
        deadline='2026-10-02T09:00:00+02:00', retry_safe=False, recovery=False, stale_seconds=900,
        placements={target: placement(row, target) for target in PLACEMENTS},
        completion=dict(argv=[sys.executable, '-m', 'experiments.run_cifar_force_mask_replay', 'collect',
                              row['case'], row['arm']], workdir=str(REPO), timeout_seconds=900, retry_safe=True))


def validate_cell(row, out):
    from experiments import reporting
    errors = reporting.validate_run(out)
    assert not errors, errors
    manifest = json.loads((out / 'manifest.json').read_text())
    metrics = json.loads((out / 'result.json').read_text())['terminal_metrics']
    config = json.loads((ROOT / 'source' / row['config']).read_text())
    assert manifest['resolved_config'] == config and manifest['environment'] == config['expected_environment']
    assert manifest['precision'] == metrics['precision'] == 'float32' and metrics['precision_checks_passed']
    assert metrics['checkpoint_sha256'] == row['checkpoint_sha256'] and metrics['examples'] == 256
    assert metrics['checkpoint_bytes_unchanged'] and metrics['parameters_and_bn_unchanged']
    assert metrics['relative_rms_targets'] == config['relative_rms_targets'] and metrics['reference_iterations'] == [6, 6, 4]
    masked = row['arm'] == 'masked'
    assert manifest.get('force_mask', 'none') == metrics.get('force_mask', 'none') == config.get('force_mask', 'none')
    assert metrics['diagnostic_blocks'] == ([1, 2, 3] if masked else [3])
    data = json.loads((out / 'artifacts/comparison.json').read_text())
    pooled, batches = EXPECTED_ROWS[row['arm']]
    assert data['layers'] == metrics['layers'] and len(data['layers']) == pooled and len(data['batches']) == batches
    assert len({(r['case_index'], r['name'], r['batch']) for r in data['batches']}) == batches
    worst_target = 0.
    for layer in data['layers']:
        error = abs(layer['output_displacement']['relative_rms'] / layer['target_relative_rms'] - 1)
        assert math.isfinite(error) and error <= .030001, (layer['name'], layer['case_index'], error)
        worst_target = max(worst_target, error)
    exactness = None
    if masked:
        checks = [r['masked_force_reference'] for r in data['batches']]
        assert all('force_mask' in r for r in data['batches'])
        exactness = dict(min_cosine=min(c['cosine'] for c in checks if c['cosine'] is not None),
                         max_norm_ratio_error=max(abs(c['norm_ratio'] - 1) for c in checks if c['norm_ratio'] is not None),
                         undefined=sum(c['cosine'] is None for c in checks))
    else:
        assert not any('force_mask' in r for r in data['batches'])
    for name, digest in config['input_sha256'].items():
        assert reporting.sha256_file(REPO / row['parent_input_root'] / name) == digest, name
    return dict(case=row['case'], arm=row['arm'], gpu=manifest['gpu'], pooled_rows=pooled, batch_rows=batches,
                max_relative_target_error=worst_target, masked_reference_exactness=exactness,
                elapsed_seconds=metrics['elapsed_seconds'], peak_memory_reserved_bytes=metrics['peak_memory_reserved_bytes'],
                comparison_sha256=reporting.sha256_file(out / 'artifacts/comparison.json'))


def collect(case_id, arm):
    from experiments import reporting
    row, = [r for r in json.loads((ROOT / 'cases.json').read_text()) if r['case'] == case_id and r['arm'] == arm]
    queue = '/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py'
    job = os.environ.get('RUN_WATCH_JOB_ID', row['job_id'])
    state = json.loads(subprocess.check_output([sys.executable, queue, 'status', job], text=True))
    target = state['attempts'][-1]['target']
    assert target in PLACEMENTS, target
    out = ROOT / 'cells' / row['id']
    if target != 'local':
        out.mkdir(parents=True, exist_ok=True)
        remote = f"{target}:{state['job']['spec']['placements'][target]['run_dir']}/"
        subprocess.run(['rsync', '-a', '-e', 'ssh -F /home/filip/.ssh/config -o BatchMode=yes', remote, str(out) + '/'],
                       check=True)
    summary = dict(validate_cell(row, out), job_id=job, target=target)
    reporting.atomic_write_json(ROOT / 'analysis' / f"{row['id']}-collection.json", summary)
    print(json.dumps(summary), flush=True)


def passing_run(grid, passes):
    """First passing target and the upper end of the contiguous passing run that starts there."""
    first = next((i for i, ok in enumerate(passes) if ok), None)
    if first is None:
        return None, None
    last = first
    while last + 1 < len(grid) and passes[last + 1]:
        last += 1
    return grid[first], grid[last]


def block_curves(data, block):
    grid = sorted({r['target_relative_rms'] for r in data['layers']})
    pooled = {t: [r for r in data['layers'] if r['block'] == block and r['target_relative_rms'] == t] for t in grid}
    batches = {t: [r for r in data['batches'] if r['block'] == block and r['target_relative_rms'] == t] for t in grid}
    worst_pooled = [min(r['local_bptt']['cosine'] for r in pooled[t]) for t in grid]
    worst_batch = [min(r['local_bptt']['cosine'] for r in batches[t]) for t in grid]
    norm_ok = [all(.9 <= r['local_bptt']['norm_ratio'] <= 1.1 for r in pooled[t]) for t in grid]
    achieved = [pooled[t][0]['output_displacement']['relative_rms'] for t in grid]
    return grid, worst_pooled, worst_batch, norm_ok, achieved


def float64_limits():
    """exp-025 full64 local-reference upper ends, same definition, for side-by-side reporting."""
    import csv
    rows = list(csv.DictReader((PARENT / 'analysis/local_bptt/layer_metrics.csv').open()))
    worst = {}
    for r in rows:
        key = (r['physical_case'], int(r['block']), float(r['target_relative_rms']))
        worst[key] = min(worst.get(key, 1.), float(r['local_bptt_cosine']))
    limits = {}
    for case, block in {(k[0], k[1]) for k in worst}:
        grid = sorted(t for c, b, t in worst if c == case and b == block)
        limits[case, block] = passing_run(grid, [worst[case, block, t] >= .99 for t in grid])
    return limits


def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from experiments import reporting
    rows = json.loads((ROOT / 'cases.json').read_text())
    collected = [r for r in rows if (ROOT / 'analysis' / f"{r['id']}-collection.json").is_file()]
    complete = len(collected) == len(rows)
    full64 = float64_limits()
    table, inactive, exactness = [], {}, []
    for row in collected:
        data = json.loads((ROOT / 'cells' / row['id'] / 'artifacts/comparison.json').read_text())
        receipt = json.loads((ROOT / 'analysis' / f"{row['id']}-collection.json").read_text())
        assert receipt['comparison_sha256'] == reporting.sha256_file(ROOT / 'cells' / row['id'] / 'artifacts/comparison.json')
        if row['arm'] == 'masked':
            exactness.append(receipt['masked_reference_exactness'])
            for b in data['batches']:
                if b['layer'] == 1 and b['case_index'] == 0:
                    inactive.setdefault((row['case'], b['block']), []).append(
                        (b['force_mask']['inactive_channels'], b['force_mask']['masked_force_square_share']))
        for block in ([1, 2, 3] if row['arm'] == 'masked' else [3]):
            grid, pooled, batch, norm, achieved = block_curves(data, block)
            strict = [p >= .99 and m >= .95 and n for p, m, n in zip(pooled, batch, norm)]
            table.append(dict(case=row['case'], arm=row['arm'], scheme=row['scheme'], epoch=row['epoch'],
                trajectory=row['trajectory'], groups=row['groups'], block=block, grid=grid,
                worst_pooled=pooled, worst_batch=batch, norm_ok=norm, achieved=achieved,
                pooled_run=passing_run(grid, [p >= .99 for p in pooled]), strict_run=passing_run(grid, strict),
                strict_pass=strict, float64_pooled_run=full64.get((row['case'], block))))
    summary = dict(complete=complete, collected=len(collected), expected=len(rows), rows=table,
                   inactive_channels={f'{c}/block{b}': dict(min=min(v[0] for v in vals), max=max(v[0] for v in vals),
                       mean_masked_share=sum(v[1] for v in vals) / len(vals)) for (c, b), vals in inactive.items()})
    if exactness:
        summary['criterion1_exactness'] = dict(min_cosine=min(e['min_cosine'] for e in exactness),
            max_norm_ratio_error=max(e['max_norm_ratio_error'] for e in exactness),
            undefined=sum(e['undefined'] for e in exactness),
            passed=all(e['min_cosine'] >= 1 - 1e-9 and e['max_norm_ratio_error'] <= 1e-9 for e in exactness))
    by = {(r['case'], r['arm'], r['block']): r for r in table}
    recovery = []
    for (case, block), vals in sorted(inactive.items()):
        if block != 3 or max(v[0] for v in vals) == 0:
            continue
        masked, control = by.get((case, 'masked', 3)), by.get((case, 'control', 3))
        if masked is None or control is None:
            continue
        upper_m, upper_c = masked['pooled_run'][1], control['pooled_run'][1]
        recovery.append(dict(case=case, masked_upper=upper_m, control_upper=upper_c,
                             passed=upper_m is not None and upper_m >= 1e-3 and (upper_c is None or upper_c < 1e-3)))
    summary['criterion2_block3_recovery'] = dict(cases=recovery, passed=bool(recovery) and all(r['passed'] for r in recovery))
    common = {}
    for group in ('eqprop', 'bptt'):
        for block in (1, 2, 3):
            members = [r for r in table if r['arm'] == 'masked' and r['block'] == block and group in r['groups']]
            if not members:
                continue
            grid = members[0]['grid']
            shared = [t for i, t in enumerate(grid) if all(m['strict_pass'][i] for m in members)]
            common[f'{group}/block{block}'] = dict(checkpoints=len(members), shared_strict_targets=shared)
    summary['common_strict_targets'] = common
    summary['criterion3_common_block3'] = dict(passed=all(common.get(f'{g}/block3', {}).get('shared_strict_targets')
                                                          for g in ('eqprop', 'bptt')))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    colors = dict(baseline='tab:blue', ours='tab:orange', legacy='tab:green')
    styles = {0: 'o', 1: 's', 5: '^', 10: 's', 30: '^', 50: 'D'}
    for gi, group in enumerate(('eqprop', 'bptt')):
        for block in (1, 2, 3):
            ax = axes[gi, block - 1]
            for r in table:
                if group not in r['groups'] or r['block'] != block:
                    continue
                masked = r['arm'] == 'masked'
                ax.plot(r['grid'], r['worst_pooled'], linestyle='-' if masked else ':', marker=styles.get(r['epoch'], 'o'),
                        markersize=3, color=colors[r['scheme']], alpha=.9 if masked else .55,
                        label=f"{r['scheme']} e{r['epoch']}{'' if masked else ' unmasked'}")
            ax.axhline(.99, color='k', lw=.6, ls='--')
            ax.set(xscale='log', ylim=(-.05, 1.02), title=f"{'EqProp' if group == 'eqprop' else 'BPTT'} checkpoints · block {block}")
            ax.grid(alpha=.2)
            if gi == 1:
                ax.set_xlabel('Target block-output relative RMS')
            if block == 1:
                ax.set_ylabel('Worst-layer pooled local cosine')
        handles, labels = axes[gi, 2].get_legend_handles_labels()
        axes[gi, 2].legend(handles, labels, fontsize=6, ncol=2, loc='lower left')
    fig.suptitle('Float32 replay: inactive-channel force mask (solid) versus unmasked block 3 (dotted)'
                 + ('' if complete else f' — partial {len(collected)}/{len(rows)}'))
    fig.tight_layout()
    (ROOT / 'analysis').mkdir(exist_ok=True)
    fig.savefig(ROOT / 'analysis/worst_layer_cosine.jpg', dpi=150)
    plt.close(fig)
    reporting.atomic_write_json(ROOT / 'analysis/summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=1, default=str)[:6000])


if __name__ == '__main__':
    mode, *args = sys.argv[1:]
    {'stage': stage, 'collect': collect, 'analyze': analyze}[mode](*args)
