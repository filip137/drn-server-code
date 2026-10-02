"""Stage, collect and analyze exp-032: initialization replays across block-output activations.

The seed-0 ours initializer (exp-025 ``init-ours-e0``) is replayed with three
block-output activations, each in float32 and float64:

* ``perfect_diode``: the original architecture (analog diode output layer);
* ``digital_relu``: linear analog output layer, rectified digitally before pooling;
* ``linear``: linear analog output layer, no rectifier.

Cohort, T/K and calibration rule are exp-025's; the RMS grid adds 0.7 and 1.0.
The runner is ``experiments.compare_cifar_block_beta_gradients``.

    python -m experiments.run_cifar_block_output_replay stage
    python -m experiments.run_cifar_block_output_replay push TARGET     # rsync inputs to a remote host
    python -m experiments.run_cifar_block_output_replay submit
    python -m experiments.run_cifar_block_output_replay collect CASE    # queue completion
    python -m experiments.run_cifar_block_output_replay analyze
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
STUDY = 'cifar-block-output-init-replay-20261002-v1'
ROOT = REPO / 'results' / STUDY
PARENT = REPO / 'results/cifar-layer-rms-all-schemes-float64-20260930-v1'
PARENT_CASE = 'init-ours-e0'
PARENT_CONFIG = REPO / 'configs/cifar/layer_rms_all_schemes_float64_20260930/init-ours-e0.json'
CONFIGS = Path('configs/cifar/block_output_init_replay_20261002')
HANDOFF = REPO / 'campaigns/cifar10-beta-selection/series/001-beta-selection/experiments/exp-032-block-output-activation-replay.md'
ACTIVATIONS = ('perfect_diode', 'digital_relu', 'linear')
PRECISIONS = ('float32', 'float64')
EXTRA_TARGETS = (0.7, 1.0)
REMOTE_RESULTS = Path('/home/filip/server_code/results')
LOCAL_PYTHON = PARENT / 'runtime/py312/bin/python'
REMOTE_PYTHON = Path('/home/filip/miniconda3/envs/py312/bin/python')
REMOTES = ('trex', 'fifi', 'loulou', 'riri')
PLACEMENTS = ('local', *REMOTES)
RUNNER_SECONDS = {'float32': 1500, 'float64': 4500}
QUEUE_SECONDS = {'float32': 1800, 'float64': 5400}
PEAK_MIB = {'float32': 11264, 'float64': 21504}
EXPECTED_SECONDS = {'float32': 600, 'float64': 2400}
INPUTS = ('checkpoint.pt', 'checkpoint_provenance.json', 'cohort.json')
ENV = dict(KMP_DISABLE_SHM='1', KMP_SHM_DISABLE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1',
           CUBLAS_WORKSPACE_CONFIG=':4096:8')
SSH = ['ssh', '-F', '/home/filip/.ssh/config', '-o', 'BatchMode=yes']


def case_ids():
    return [f'{activation}-{precision}' for activation in ACTIVATIONS for precision in PRECISIONS]


def split_case(case):
    activation, precision = case.rsplit('-', 1)
    assert activation in ACTIVATIONS and precision in PRECISIONS, case
    return activation, precision


def job_id(case):
    return f'cifar-block-output-init-{case}-20261002'


def source_files():
    for directory in ('experiments', 'labs', 'model', 'training'):
        for path in sorted((REPO / directory).rglob('*.py')):
            if '__pycache__' not in path.parts and 'tests' not in path.relative_to(REPO).parts:
                yield path.relative_to(REPO)


def stage_inputs(activation):
    """One input root per activation; the checkpoint is hard-linked, the model config differs only here."""
    from experiments import reporting
    parent = PARENT / 'cases' / PARENT_CASE / 'inputs'
    target = ROOT / 'cases' / activation / 'inputs'
    target.mkdir(parents=True)
    for name in INPUTS:
        if name == 'checkpoint.pt':
            os.link(parent / name, target / name)
        else:
            shutil.copy2(parent / name, target / name)
    config = json.loads((parent / 'model_config.json').read_text())
    assert 'block_output_activation' not in config, 'Parent config must use the implicit original outputs'
    config['block_output_activation'] = activation
    reporting.atomic_write_json(target / 'model_config.json', config)
    return {f'inputs/{name}': reporting.sha256_file(target / name) for name in (*INPUTS, 'model_config.json')}


def make_config(case, input_sha256):
    activation, precision = split_case(case)
    config = json.loads(PARENT_CONFIG.read_text())
    assert config['case_id'] == PARENT_CASE and config['precision'] == 'float64'
    targets = [*config['relative_rms_targets'], *EXTRA_TARGETS]
    assert targets == sorted(targets)
    config.update(study_id=STUDY, case_id=case, block_output_activation=activation, precision=precision,
                  budget_seconds=RUNNER_SECONDS[precision], relative_rms_targets=targets,
                  input_sha256=input_sha256)
    return config


def stage():
    from experiments import reporting
    assert not ROOT.exists(), 'Study already staged'
    (REPO / CONFIGS).mkdir(parents=True, exist_ok=True)
    hashes = {activation: stage_inputs(activation) for activation in ACTIVATIONS}
    rows = []
    for case in case_ids():
        activation, precision = split_case(case)
        path = REPO / CONFIGS / f'{case}.json'
        assert not path.exists(), path
        reporting.atomic_write_json(path, make_config(case, hashes[activation]))
        rows.append(dict(id=case, activation=activation, precision=precision, input_root=f'cases/{activation}',
                         config=str(CONFIGS / f'{case}.json'), job_id=job_id(case),
                         checkpoint_sha256=hashes[activation]['inputs/checkpoint.pt']))
    source = ROOT / 'source'
    files = [*source_files(), *sorted(p.relative_to(REPO) for p in (REPO / CONFIGS).glob('*.json'))]
    for relative in files:
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, source / relative)
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip())
    identity = dict(commit=commit, dirty=dirty, study_id=STUDY,
        note=('exp-032 initialization replay of the ours seed-0 initializer with perfect_diode, digital_relu '
              'and linear block outputs, float32 and float64. Uncommitted files are identified by hash.'),
        source_sha256={str(r): reporting.sha256_file(source / r) for r in files})
    reporting.atomic_write_json(source / 'source_identity.json', identity)
    reporting.atomic_write_json(ROOT / 'cases.json', rows)
    for row in rows:
        reporting.atomic_write_json(ROOT / 'launch' / f"{row['id']}.json", queue_spec(row))
    print(json.dumps(dict(staged=len(rows), source_files=len(files)), indent=1))


def push(target):
    """Copy frozen source and input roots to a remote host, then verify their hashes there."""
    assert target in REMOTES, target
    remote = REMOTE_RESULTS / STUDY
    subprocess.run([*SSH, target, f'mkdir -p {remote}'], check=True)
    subprocess.run(['rsync', '-aH', '-e', ' '.join(SSH), f'{ROOT}/source', f'{ROOT}/cases', f'{ROOT}/cases.json',
                    f'{target}:{remote}/'], check=True)
    from experiments import reporting
    expected = {str(Path('source/source_identity.json')): reporting.sha256_file(ROOT / 'source/source_identity.json')}
    for row in json.loads((ROOT / 'cases.json').read_text()):
        for name, digest in json.loads((ROOT / 'source' / row['config']).read_text())['input_sha256'].items():
            expected[f"{row['input_root']}/{name}"] = digest
    listing = subprocess.check_output([*SSH, target, 'cd ' + str(remote) + ' && sha256sum ' + ' '.join(sorted(expected))],
                                      text=True)
    observed = {line.split()[1]: line.split()[0] for line in listing.splitlines()}
    assert observed == expected, dict(target=target, observed=observed, expected=expected)
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
                        dict(path=str(config), sha256=reporting.sha256_file(ROOT / 'source' / row['config'])),
                        dict(path=str(input_root / 'inputs/checkpoint.pt'), sha256=row['checkpoint_sha256'])])


def queue_spec(row, targets=PLACEMENTS):
    precision = row['precision']
    return dict(id=row['job_id'], project_dir=str(REPO), handoff=str(HANDOFF),
        authorization=('Filip asked (October 2) to test linear block-output layers with and without a digital '
                       'ReLU: read-only initialization replays against the local BPTT reference, float32 and '
                       'float64, plus the original diode-output control. No training, noise or test split.'),
        resource_profile='default', peak_memory_mib=PEAK_MIB[precision], headroom_mib=2048,
        priority=4 if precision == 'float32' else 3, expected_seconds=EXPECTED_SECONDS[precision],
        budget_seconds=QUEUE_SECONDS[precision], deadline='2026-10-02T22:00:00+02:00',
        retry_safe=False, recovery=False, stale_seconds=900,
        placements={target: placement(row, target) for target in targets},
        completion=dict(argv=[sys.executable, '-m', 'experiments.run_cifar_block_output_replay', 'collect',
                              row['id']], workdir=str(REPO), timeout_seconds=900, retry_safe=True))


def submit(*targets):
    """Submit every staged case with the placements whose inputs have been pushed (local is always staged)."""
    targets = targets or PLACEMENTS
    queue = '/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py'
    for row in json.loads((ROOT / 'cases.json').read_text()):
        from experiments import reporting
        path = ROOT / 'launch' / f"{row['id']}.json"
        reporting.atomic_write_json(path, queue_spec(row, targets))
        subprocess.run([sys.executable, queue, 'submit', str(path)], check=True)


def validate_cell(row, out):
    from experiments import reporting
    errors = reporting.validate_run(out)
    assert not errors, errors
    manifest = json.loads((out / 'manifest.json').read_text())
    metrics = json.loads((out / 'result.json').read_text())['terminal_metrics']
    config = json.loads((ROOT / 'source' / row['config']).read_text())
    model_config = json.loads((ROOT / row['input_root'] / 'inputs/model_config.json').read_text())
    assert model_config['block_output_activation'] == row['activation'] == config['block_output_activation']
    assert manifest['resolved_config'] == config and manifest['environment'] == config['expected_environment']
    assert manifest['precision'] == metrics['precision'] == row['precision'] and metrics['precision_checks_passed']
    assert metrics['checkpoint_sha256'] == row['checkpoint_sha256'] and metrics['examples'] == 256
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
    return dict(case=row['id'], activation=row['activation'], precision=row['precision'], gpu=manifest['gpu'],
                pooled_rows=pooled, max_relative_target_error=worst_target,
                elapsed_seconds=metrics['elapsed_seconds'], peak_memory_reserved_bytes=metrics['peak_memory_reserved_bytes'],
                comparison_sha256=reporting.sha256_file(out / 'artifacts/comparison.json'))


def collect(case):
    from experiments import reporting
    row, = [r for r in json.loads((ROOT / 'cases.json').read_text()) if r['id'] == case]
    queue = '/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py'
    job = os.environ.get('RUN_WATCH_JOB_ID', row['job_id'])
    state = json.loads(subprocess.check_output([sys.executable, queue, 'status', job], text=True))
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
    """First passing target and the upper end of the contiguous passing run that starts there."""
    first = next((i for i, ok in enumerate(passes) if ok), None)
    if first is None:
        return None, None
    last = first
    while last + 1 < len(grid) and passes[last + 1]:
        last += 1
    return grid[first], grid[last]


def block_curves(data, block, reference='local_bptt'):
    grid = sorted({r['target_relative_rms'] for r in data['layers']})
    pooled = {t: [r for r in data['layers'] if r['block'] == block and r['target_relative_rms'] == t] for t in grid}
    batches = {t: [r for r in data['batches'] if r['block'] == block and r['target_relative_rms'] == t] for t in grid}
    worst_pooled = [min(r[reference]['cosine'] for r in pooled[t]) for t in grid]
    worst_batch = [min(r[reference]['cosine'] for r in batches[t]) for t in grid]
    norm_ok = [all(.9 <= r[reference]['norm_ratio'] <= 1.1 for r in pooled[t]) for t in grid]
    return grid, worst_pooled, worst_batch, norm_ok


def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from experiments import reporting
    rows = json.loads((ROOT / 'cases.json').read_text())
    collected = [r for r in rows if (ROOT / 'analysis' / f"{r['id']}-collection.json").is_file()]
    table = []
    for row in collected:
        out = ROOT / 'cells' / row['id']
        receipt = json.loads((ROOT / 'analysis' / f"{row['id']}-collection.json").read_text())
        assert receipt['comparison_sha256'] == reporting.sha256_file(out / 'artifacts/comparison.json')
        data = json.loads((out / 'artifacts/comparison.json').read_text())
        for block in (1, 2, 3):
            grid, pooled, batch, norm = block_curves(data, block)
            strict = [p >= .99 and m >= .95 and n for p, m, n in zip(pooled, batch, norm)]
            _, native_pooled, _, _ = block_curves(data, block, 'native_bptt')
            table.append(dict(case=row['id'], activation=row['activation'], precision=row['precision'], block=block,
                grid=grid, worst_pooled=pooled, worst_batch=batch, norm_ok=norm, strict_pass=strict,
                worst_pooled_native=native_pooled,
                pooled_run=passing_run(grid, [p >= .99 for p in pooled]), strict_run=passing_run(grid, strict)))
    summary = dict(complete=len(collected) == len(rows), collected=len(collected), expected=len(rows),
                   windows={f"{r['case']}/block{r['block']}": dict(pooled_run=r['pooled_run'], strict_run=r['strict_run'],
                            best_worst_pooled=max(r['worst_pooled'])) for r in table}, rows=table)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True, sharey=True)
    colors = dict(perfect_diode='tab:gray', digital_relu='tab:orange', linear='tab:blue')
    labels = dict(perfect_diode='analog diode output (original)', digital_relu='linear analog + digital ReLU',
                  linear='linear analog, no rectifier')
    for pi, precision in enumerate(PRECISIONS):
        for block in (1, 2, 3):
            ax = axes[pi, block - 1]
            for r in table:
                if r['precision'] == precision and r['block'] == block:
                    ax.plot(r['grid'], r['worst_pooled'], marker='o', markersize=3, color=colors[r['activation']],
                            label=labels[r['activation']])
            ax.axhline(.99, color='k', lw=.6, ls='--')
            ax.set(xscale='log', ylim=(-.05, 1.02), title=f'{precision} · block {block}')
            ax.grid(alpha=.2)
            if pi == 1:
                ax.set_xlabel('Target block-output relative RMS')
            if block == 1:
                ax.set_ylabel('Worst-layer pooled cosine vs local BPTT')
        axes[pi, 2].legend(fontsize=7, loc='lower left')
    fig.suptitle('Ours initialization: EqProp gradient agreement versus output displacement by block-output activation'
                 + ('' if summary['complete'] else f" — partial {len(collected)}/{len(rows)}"))
    fig.tight_layout()
    (ROOT / 'analysis').mkdir(exist_ok=True)
    fig.savefig(ROOT / 'analysis/worst_layer_cosine.jpg', dpi=150)
    plt.close(fig)
    reporting.atomic_write_json(ROOT / 'analysis/summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=1, default=str)[:8000])


if __name__ == '__main__':
    mode, *args = sys.argv[1:]
    {'stage': stage, 'push': push, 'submit': submit, 'collect': collect, 'analyze': analyze}[mode](*args)
