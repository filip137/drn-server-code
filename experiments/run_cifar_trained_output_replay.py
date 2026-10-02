"""Stage, collect and analyze exp-033: float32 replays of trained checkpoints with linear block outputs.

Two groups, both replayed with ``experiments.compare_cifar_block_beta_gradients``
in float32 on exp-025's cohort, T/K and calibration rule (grid plus 0.7 and 1.0):

* ``transplant``: the 15 trained exp-025 checkpoints (diode-output architecture,
  EqProp epochs 1/5 and BPTT epochs 10/30/50 for baseline/ours/legacy) loaded
  into the ``digital_relu`` architecture. Matched float32 diode controls are
  exp-030's unmasked replays of the same checkpoints.
* ``linear``: the Codex linear-output BPTT checkpoints (ours/legacy, epochs
  1/5/10) replayed in the ``linear`` architecture they were trained with.

    python -m experiments.run_cifar_trained_output_replay stage-transplant
    python -m experiments.run_cifar_trained_output_replay stage-linear SCHEME EPOCH
    python -m experiments.run_cifar_trained_output_replay push TARGET
    python -m experiments.run_cifar_trained_output_replay submit CASE [CASE ...]
    python -m experiments.run_cifar_trained_output_replay collect CASE    # queue completion
    python -m experiments.run_cifar_trained_output_replay analyze
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
STUDY = 'cifar-trained-output-replay-20261002-v1'
ROOT = REPO / 'results' / STUDY
PARENT = REPO / 'results/cifar-layer-rms-all-schemes-float64-20260930-v1'
PARENT_CONFIGS = REPO / 'configs/cifar/layer_rms_all_schemes_float64_20260930'
CONTROL = REPO / 'results/cifar-force-mask-float32-replay-20261001-v1'
LINEAR = REPO / 'results/cifar10-l8-linear-outputs-bptt-20261002-v1'
LINEAR_REMOTE = {'ours': 'fifi', 'legacy': 'local'}
CONFIGS = Path('configs/cifar/trained_output_replay_20261002')
HANDOFF = REPO / 'campaigns/cifar10-beta-selection/series/001-beta-selection/experiments/exp-033-trained-linear-output-replay.md'
EXTRA_TARGETS = (0.7, 1.0)
REMOTE_RESULTS = Path('/home/filip/server_code/results')
LOCAL_PYTHON = PARENT / 'runtime/py312/bin/python'
REMOTE_PYTHON = Path('/home/filip/miniconda3/envs/py312/bin/python')
REMOTES = ('trex', 'fifi', 'loulou', 'riri')
PLACEMENTS = ('local', *REMOTES)
RUNNER_SECONDS, QUEUE_SECONDS, EXPECTED_SECONDS = 1500, 1800, 600
INPUTS = ('checkpoint_provenance.json', 'cohort.json')
ENV = dict(KMP_DISABLE_SHM='1', KMP_SHM_DISABLE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1',
           CUBLAS_WORKSPACE_CONFIG=':4096:8')
SSH = ['ssh', '-F', '/home/filip/.ssh/config', '-o', 'BatchMode=yes']
QUEUE = '/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py'


def rows():
    path = ROOT / 'cases.json'
    return json.loads(path.read_text()) if path.is_file() else []


def save_rows(table):
    from experiments import reporting
    reporting.atomic_write_json(ROOT / 'cases.json', table)


def job_id(case):
    return f'cifar-trained-output-{case}-20261002'


def source_files():
    for directory in ('experiments', 'labs', 'model', 'training'):
        for path in sorted((REPO / directory).rglob('*.py')):
            if '__pycache__' not in path.parts and 'tests' not in path.relative_to(REPO).parts:
                yield path.relative_to(REPO)


def freeze_source():
    """Code identity only; each case's config is identified by hash in cases.json and its queue spec."""
    from experiments import reporting
    source = ROOT / 'source'
    assert not source.exists(), 'Frozen source already staged'
    files = list(source_files())
    for relative in files:
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, source / relative)
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip())
    reporting.atomic_write_json(source / 'source_identity.json', dict(commit=commit, dirty=dirty, study_id=STUDY,
        note=('exp-033 float32 replays: exp-025 trained checkpoints in the digital_relu architecture and '
              'Codex linear-output BPTT checkpoints in the linear architecture. Configs are hashed per case.'),
        source_sha256={str(r): reporting.sha256_file(source / r) for r in files}))


def write_case(case, model_config, checkpoint, base_config, activation, extra, provenance=None):
    """Input root plus runner config for one case; returns its cases.json row."""
    from experiments import reporting
    inputs = ROOT / 'cases' / case / 'inputs'
    inputs.mkdir(parents=True)
    os.link(checkpoint, inputs / 'checkpoint.pt')
    parent_inputs = PARENT / 'cases/init-ours-e0/inputs'
    for name in INPUTS:
        if name == 'checkpoint_provenance.json' and provenance is not None:
            reporting.atomic_write_json(inputs / name, provenance)
        else:
            shutil.copy2(extra.get(name, parent_inputs / name), inputs / name)
    assert model_config.get('block_output_activation', 'perfect_diode') in ('perfect_diode', activation)
    model_config = dict(model_config, block_output_activation=activation)
    reporting.atomic_write_json(inputs / 'model_config.json', model_config)
    digests = {f'inputs/{n}': reporting.sha256_file(inputs / n)
               for n in ('checkpoint.pt', *INPUTS, 'model_config.json')}
    config = dict(base_config)
    targets = [*config['relative_rms_targets'], *EXTRA_TARGETS]
    assert targets == sorted(targets)
    config.update(study_id=STUDY, case_id=case, block_output_activation=activation, precision='float32',
                  budget_seconds=RUNNER_SECONDS, relative_rms_targets=targets, input_sha256=digests,
                  checkpoint_file='inputs/checkpoint.pt')
    config.pop('force_mask', None)
    config.pop('diagnostic_blocks', None)
    for directory in (REPO / CONFIGS, ROOT / 'source' / CONFIGS):
        directory.mkdir(parents=True, exist_ok=True)
        assert not (directory / f'{case}.json').exists(), case
        reporting.atomic_write_json(directory / f'{case}.json', config)
    return dict(id=case, activation=activation, input_root=f'cases/{case}', config=str(CONFIGS / f'{case}.json'),
                config_sha256=reporting.sha256_file(ROOT / 'source' / CONFIGS / f'{case}.json'),
                job_id=job_id(case), checkpoint_sha256=digests['inputs/checkpoint.pt'])


def stage_transplant():
    freeze_source()
    table = []
    for parent in json.loads((PARENT / 'cases.json').read_text()):
        if parent['epoch'] == 0:
            continue
        base = json.loads((PARENT_CONFIGS / f"{parent['id']}.json").read_text())
        assert base['case_id'] == parent['id'] and base['checkpoint_epoch'] == parent['epoch']
        inputs = PARENT / parent['input_root'] / 'inputs'
        model_config = json.loads((inputs / 'model_config.json').read_text())
        row = write_case(f"transplant-{parent['id']}", model_config, inputs / 'checkpoint.pt', base,
                         'digital_relu', {n: inputs / n for n in INPUTS})
        row.update(group='transplant', scheme=parent['scheme'], epoch=parent['epoch'],
                   trajectory=parent['trajectory'], control_case=parent['id'])
        assert row['checkpoint_sha256'] == parent['checkpoint_sha256']
        table.append(row)
    save_rows(table)
    print(json.dumps(dict(staged=len(table))))


def stage_linear(scheme, epoch):
    """Stage one Codex linear-output checkpoint after it has been written (pulled from its host if remote)."""
    import torch
    epoch = int(epoch)
    case = f'linear-{scheme}-e{epoch}'
    assert scheme in LINEAR_REMOTE and epoch in (1, 5, 10), (scheme, epoch)
    assert case not in {r['id'] for r in rows()}, case
    source = LINEAR / 'cells' / scheme / 'checkpoints' / f'epoch_{epoch:03d}.pt'
    if LINEAR_REMOTE[scheme] != 'local':
        source.parent.mkdir(parents=True, exist_ok=True)
        remote = REMOTE_RESULTS / LINEAR.name / 'cells' / scheme / 'checkpoints' / source.name
        subprocess.run(['rsync', '-a', '-e', ' '.join(SSH), f'{LINEAR_REMOTE[scheme]}:{remote}', str(source)], check=True)
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == epoch and checkpoint['config']['scheme'] == scheme
    assert checkpoint['config']['block_output_activation'] == 'linear'
    model_config = checkpoint['config']
    del checkpoint
    base = json.loads((PARENT_CONFIGS / f'bptt-{scheme}-e10.json').read_text())
    base.update(checkpoint_epoch=epoch, scheme=scheme, trajectory='bptt-linear-outputs')
    from experiments import reporting
    provenance = dict(source_study=LINEAR.name, source_checkpoint=str(source.relative_to(REPO)),
                      source_sha256=reporting.sha256_file(source), epoch=epoch, scheme=scheme,
                      block_output_activation='linear', trainer='experiments.train_cifar_l8_analog (BPTT)')
    row = write_case(case, model_config, source, base, 'linear', {}, provenance)
    row.update(group='linear', scheme=scheme, epoch=epoch, trajectory='bptt-linear-outputs', control_case=None,
               source_checkpoint=str(source.relative_to(REPO)))
    save_rows([*rows(), row])
    print(json.dumps(dict(staged=case)))


def push(target):
    """Copy frozen source and every staged input root to a remote host, then verify hashes there."""
    from experiments import reporting
    assert target in REMOTES, target
    remote = REMOTE_RESULTS / STUDY
    subprocess.run([*SSH, target, f'mkdir -p {remote}'], check=True)
    subprocess.run(['rsync', '-aH', '-e', ' '.join(SSH), f'{ROOT}/source', f'{ROOT}/cases', f'{ROOT}/cases.json',
                    f'{target}:{remote}/'], check=True)
    expected = {'source/source_identity.json': reporting.sha256_file(ROOT / 'source/source_identity.json')}
    for row in rows():
        expected[f"source/{row['config']}"] = row['config_sha256']
        for name, digest in json.loads((ROOT / 'source' / row['config']).read_text())['input_sha256'].items():
            expected[f"{row['input_root']}/{name}"] = digest
    listing = subprocess.check_output([*SSH, target, f'cd {remote} && sha256sum ' + ' '.join(sorted(expected))], text=True)
    observed = {line.split()[1]: line.split()[0] for line in listing.splitlines()}
    assert observed == expected, dict(target=target, missing=sorted(set(expected) - set(observed)))
    print(json.dumps(dict(target=target, verified_files=len(observed))))


def placement(row, target):
    from experiments import reporting
    local = target == 'local'
    base = ROOT if local else REMOTE_RESULTS / STUDY
    python = LOCAL_PYTHON if local else REMOTE_PYTHON
    dataset = (REPO if local else REMOTE_RESULTS.parent) / 'results/cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1/data/cifar-10-batches-py'
    input_root = base / row['input_root']
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
                        dict(path=str(config), sha256=row['config_sha256']),
                        dict(path=str(input_root / 'inputs/checkpoint.pt'), sha256=row['checkpoint_sha256'])])


def queue_spec(row):
    return dict(id=row['job_id'], project_dir=str(REPO), handoff=str(HANDOFF),
        authorization=('Filip asked (October 2) for float32 replays of trained checkpoints: Codex linear-output '
                       'BPTT checkpoints in the linear architecture and the earlier diode-output checkpoints '
                       'loaded into the digital_relu architecture. Read-only; no training, noise or test split.'),
        resource_profile='default', peak_memory_mib=11264, headroom_mib=2048, priority=4,
        expected_seconds=EXPECTED_SECONDS, budget_seconds=QUEUE_SECONDS, deadline='2026-10-03T08:00:00+02:00',
        retry_safe=False, recovery=False, stale_seconds=900,
        placements={target: placement(row, target) for target in PLACEMENTS},
        completion=dict(argv=[sys.executable, '-m', 'experiments.run_cifar_trained_output_replay', 'collect',
                              row['id']], workdir=str(REPO), timeout_seconds=900, retry_safe=True))


def submit(*cases):
    from experiments import reporting
    table = {r['id']: r for r in rows()}
    for case in cases:
        row = table[case]
        path = ROOT / 'launch' / f'{case}.json'
        reporting.atomic_write_json(path, queue_spec(row))
        subprocess.run([sys.executable, QUEUE, 'submit', str(path)], check=True)


def validate_cell(row, out):
    from experiments import reporting
    errors = reporting.validate_run(out)
    assert not errors, errors
    manifest = json.loads((out / 'manifest.json').read_text())
    metrics = json.loads((out / 'result.json').read_text())['terminal_metrics']
    config = json.loads((ROOT / 'source' / row['config']).read_text())
    model_config = json.loads((ROOT / row['input_root'] / 'inputs/model_config.json').read_text())
    assert model_config['block_output_activation'] == row['activation'] == config['block_output_activation']
    assert reporting.sha256_file(ROOT / 'source' / row['config']) == row['config_sha256']
    assert manifest['resolved_config'] == config and manifest['environment'] == config['expected_environment']
    assert manifest['precision'] == metrics['precision'] == 'float32' and metrics['precision_checks_passed']
    assert metrics['checkpoint_sha256'] == row['checkpoint_sha256'] and metrics['examples'] == 256
    assert metrics['checkpoint_epoch'] == row['epoch']
    assert metrics['checkpoint_bytes_unchanged'] and metrics['parameters_and_bn_unchanged']
    assert metrics['relative_rms_targets'] == config['relative_rms_targets'] and metrics['reference_iterations'] == [6, 6, 4]
    assert metrics['diagnostic_blocks'] == [1, 2, 3] and 'force_mask' not in metrics
    data = json.loads((out / 'artifacts/comparison.json').read_text())
    pooled = 8 * len(config['relative_rms_targets'])
    assert data['layers'] == metrics['layers'] and len(data['layers']) == pooled and len(data['batches']) == pooled * 8
    worst_target = 0.
    for layer in data['layers']:
        error = abs(layer['output_displacement']['relative_rms'] / layer['target_relative_rms'] - 1)
        assert math.isfinite(error) and error <= .030001, (layer['name'], layer['case_index'], error)
        worst_target = max(worst_target, error)
    for name, digest in config['input_sha256'].items():
        assert reporting.sha256_file(ROOT / row['input_root'] / name) == digest, name
    return dict(case=row['id'], group=row['group'], gpu=manifest['gpu'], pooled_rows=pooled,
                max_relative_target_error=worst_target, elapsed_seconds=metrics['elapsed_seconds'],
                peak_memory_reserved_bytes=metrics['peak_memory_reserved_bytes'],
                comparison_sha256=reporting.sha256_file(out / 'artifacts/comparison.json'))


def collect(case):
    from experiments import reporting
    row, = [r for r in rows() if r['id'] == case]
    job = os.environ.get('RUN_WATCH_JOB_ID', row['job_id'])
    state = json.loads(subprocess.check_output([sys.executable, QUEUE, 'status', job], text=True))
    target = state['attempts'][-1]['target']
    assert target in PLACEMENTS, target
    out = ROOT / 'cells' / row['id']
    if target != 'local':
        out.mkdir(parents=True, exist_ok=True)
        remote = f"{target}:{state['job']['spec']['placements'][target]['run_dir']}/"
        subprocess.run(['rsync', '-a', '-e', ' '.join(SSH), remote, str(out) + '/'], check=True)
    summary = dict(validate_cell(row, out), job_id=job, target=target)
    reporting.atomic_write_json(ROOT / 'analysis' / f"{row['id']}-collection.json", summary)
    print(json.dumps(summary), flush=True)


def passing_run(grid, passes):
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
    return grid, worst_pooled, worst_batch, norm_ok


def curves_row(data, block):
    grid, pooled, batch, norm = block_curves(data, block)
    strict = [p >= .99 and m >= .95 and n for p, m, n in zip(pooled, batch, norm)]
    return dict(block=block, grid=grid, worst_pooled=pooled, worst_batch=batch, strict_pass=strict,
                pooled_run=passing_run(grid, [p >= .99 for p in pooled]), strict_run=passing_run(grid, strict))


def control_curves(case):
    """exp-030 unmasked float32 diode replay: block3 from the control arm, blocks1/2 from the masked arm.

    exp-030 recorded no inactive channels in blocks 1/2 at any checkpoint, so its masked
    arm is the unmasked replay there; this is asserted per batch.
    """
    out = {}
    for arm in ('control', 'masked'):
        data = json.loads((CONTROL / 'cells' / f'{case}-{arm}' / 'artifacts/comparison.json').read_text())
        blocks = [3] if arm == 'control' else [1, 2]
        if arm == 'masked':
            assert all(b['force_mask']['inactive_channels'] == 0 for b in data['batches'] if b['block'] in blocks), case
        for block in blocks:
            out[block] = curves_row(data, block)
    return out


def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from experiments import reporting
    table = rows()
    collected = [r for r in table if (ROOT / 'analysis' / f"{r['id']}-collection.json").is_file()]
    results = []
    for row in collected:
        out = ROOT / 'cells' / row['id']
        receipt = json.loads((ROOT / 'analysis' / f"{row['id']}-collection.json").read_text())
        assert receipt['comparison_sha256'] == reporting.sha256_file(out / 'artifacts/comparison.json')
        data = json.loads((out / 'artifacts/comparison.json').read_text())
        control = control_curves(row['control_case']) if row['control_case'] else {}
        for block in (1, 2, 3):
            entry = dict(case=row['id'], group=row['group'], scheme=row['scheme'], epoch=row['epoch'],
                         trajectory=row['trajectory'], **curves_row(data, block))
            if block in control:
                entry['control'] = {k: control[block][k] for k in ('grid', 'worst_pooled', 'pooled_run', 'strict_run')}
            results.append(entry)
    summary = dict(collected=len(collected), expected=len(table), complete=len(collected) == len(table),
                   windows={f"{r['case']}/block{r['block']}": dict(
                       pooled_run=r['pooled_run'], strict_run=r['strict_run'],
                       control_pooled_run=r.get('control', {}).get('pooled_run')) for r in results},
                   rows=results)
    groups = [g for g in ('transplant', 'linear') if any(r['group'] == g for r in results)]
    if groups:
        fig, axes = plt.subplots(len(groups), 3, figsize=(15, 4.4 * len(groups)), sharex=True, sharey=True,
                                 squeeze=False)
        colors = dict(baseline='tab:blue', ours='tab:orange', legacy='tab:green')
        markers = {1: 'o', 5: 's', 10: '^', 30: 'v', 50: 'D'}
        for gi, group in enumerate(groups):
            for block in (1, 2, 3):
                ax = axes[gi, block - 1]
                for r in results:
                    if r['group'] != group or r['block'] != block:
                        continue
                    label = f"{r['scheme']} {r['trajectory'].split('-')[0]} e{r['epoch']}"
                    ax.plot(r['grid'], r['worst_pooled'], marker=markers.get(r['epoch'], 'o'), markersize=3,
                            color=colors[r['scheme']], alpha=.9, label=label)
                    if 'control' in r:
                        ax.plot(r['control']['grid'], r['control']['worst_pooled'], ls=':', lw=.9,
                                color=colors[r['scheme']], alpha=.5)
                ax.axhline(.99, color='k', lw=.6, ls='--')
                title = ('diode-trained weights, digital ReLU outputs (dotted: diode outputs, exp-030)'
                         if group == 'transplant' else 'linear-output BPTT checkpoints (Codex)')
                ax.set(xscale='log', ylim=(-.05, 1.02), title=f'block {block}: {title}' if block == 1 else f'block {block}')
                ax.grid(alpha=.2)
                if gi == len(groups) - 1:
                    ax.set_xlabel('Target block-output relative RMS')
                if block == 1:
                    ax.set_ylabel('Worst-layer pooled cosine vs local BPTT')
            axes[gi, 2].legend(fontsize=6, ncol=2, loc='lower left')
        fig.suptitle('Float32 trained-checkpoint replays with linear block outputs'
                     + ('' if summary['complete'] else f" — partial {len(collected)}/{len(table)}"))
        fig.tight_layout()
        (ROOT / 'analysis').mkdir(exist_ok=True)
        fig.savefig(ROOT / 'analysis/worst_layer_cosine.jpg', dpi=150)
        plt.close(fig)
    reporting.atomic_write_json(ROOT / 'analysis/summary.json', summary)
    print(json.dumps(summary['windows'], indent=1, default=str)[:8000])


if __name__ == '__main__':
    mode, *args = sys.argv[1:]
    {'stage-transplant': stage_transplant, 'stage-linear': stage_linear, 'push': push, 'submit': submit,
     'collect': collect, 'analyze': analyze}[mode](*args)
