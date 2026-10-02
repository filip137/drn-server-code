"""Stage, smoke, submit and collect exp-034: BPTT training of analog L8 with digital-ReLU block outputs.

The contract is the Codex linear-output pilot's ours arm
(``configs/cifar/linear_outputs_bptt_20261002/ours.json``) with only
``block_output_activation`` changed to ``digital_relu`` and more checkpoint
epochs, so its trajectory can be replayed for beta selection.

    python -m experiments.run_cifar_digital_relu_bptt stage
    python -m experiments.run_cifar_digital_relu_bptt push TARGET
    python -m experiments.run_cifar_digital_relu_bptt submit TARGET
    python -m experiments.run_cifar_digital_relu_bptt collect           # queue completion
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
STUDY = 'cifar10-l8-digital-relu-bptt-20261002-v1'
ROOT = REPO / 'results' / STUDY
PARENT_CONFIG = REPO / 'configs/cifar/linear_outputs_bptt_20261002/ours.json'
CONFIG = Path('configs/cifar/digital_relu_outputs_bptt_20261002/ours.json')
HANDOFF = REPO / 'campaigns/cifar10-beta-selection/series/001-beta-selection/experiments/exp-034-digital-relu-bptt-trajectory.md'
CHECKPOINT_EPOCHS = [1, 2, 3, 5, 7, 10]
REMOTE_RESULTS = Path('/home/filip/server_code/results')
LOCAL_PYTHON = REPO / 'results/cifar-layer-rms-all-schemes-float64-20260930-v1/runtime/py312/bin/python'
REMOTE_PYTHON = Path('/home/filip/miniconda3/envs/py312/bin/python')
DATASET = 'results/cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1/data/cifar-10-batches-py'
QUEUE = '/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py'
SSH = ['ssh', '-F', '/home/filip/.ssh/config', '-o', 'BatchMode=yes']
ENV = dict(KMP_DISABLE_SHM='1', KMP_SHM_DISABLE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
           PYTHONUNBUFFERED='1')
# r2: resubmitted at lower priority after the first attempt was cancelled to put linear replays first.
JOB = 'cifar-digital-relu-bptt-ours-r2-20261002'
RUN_CELL = 'ours-r2'
BUDGET_SECONDS = 21590


def make_config():
    config = json.loads(PARENT_CONFIG.read_text())
    assert config['block_output_activation'] == 'linear' and config['scheme'] == 'ours'
    config.update(block_output_activation='digital_relu', arm_id='ours-digital-relu-outputs-bptt-e10',
                  study_id=STUDY, evidence_class='cifar10_analog_digital_relu_outputs_bptt_trajectory',
                  checkpoint_epochs=CHECKPOINT_EPOCHS)
    return config


def source_files():
    for directory in ('experiments', 'labs', 'model', 'training'):
        for path in sorted((REPO / directory).rglob('*.py')):
            if '__pycache__' not in path.parts and 'tests' not in path.relative_to(REPO).parts:
                yield path.relative_to(REPO)


def stage():
    """Freeze source and config, then prepare the seed-0 initializer on CPU (as the Codex pilot did)."""
    from experiments import reporting
    assert not ROOT.exists(), 'Study already staged'
    (REPO / CONFIG).parent.mkdir(parents=True, exist_ok=True)
    assert not (REPO / CONFIG).exists()
    reporting.atomic_write_json(REPO / CONFIG, make_config())
    source = ROOT / 'source'
    files = [*source_files(), CONFIG]
    for relative in files:
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, source / relative)
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = bool(subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip())
    reporting.atomic_write_json(source / 'source_identity.json', dict(commit=commit, dirty=dirty, study_id=STUDY,
        note='Ours analog L8, linear analog block outputs with digital ReLU, BPTT, zero decay, seed 0.',
        source_sha256={str(r): reporting.sha256_file(source / r) for r in files}))
    subprocess.run([str(LOCAL_PYTHON), '-m', 'experiments.train_cifar_l8_analog', str(CONFIG),
                    '--mode', 'prepare', '--output-dir', '../assets/ours', '--device', 'cpu',
                    '--target', 'local-preparation', '--dataset-root', str(REPO / DATASET),
                    '--source-identity', str(source / 'source_identity.json')],
                   cwd=source, env=dict(os.environ, **ENV, CUDA_VISIBLE_DEVICES=''), check=True)
    print(json.dumps(dict(staged=STUDY, initial_model=reporting.sha256_file(
        ROOT / 'assets/ours/checkpoints/initial_model.pt'))))


def push(target):
    from experiments import reporting
    remote = REMOTE_RESULTS / STUDY
    subprocess.run([*SSH, target, f'mkdir -p {remote}'], check=True)
    subprocess.run(['rsync', '-a', '-e', ' '.join(SSH), f'{ROOT}/source', f'{ROOT}/assets', f'{target}:{remote}/'],
                   check=True)
    expected = {name: reporting.sha256_file(ROOT / name) for name in
                ('source/source_identity.json', f'source/{CONFIG}', 'assets/ours/checkpoints/initial_model.pt')}
    listing = subprocess.check_output([*SSH, target, f'cd {remote} && sha256sum ' + ' '.join(sorted(expected))], text=True)
    observed = {line.split()[1]: line.split()[0] for line in listing.splitlines()}
    assert observed == expected, dict(observed=observed, expected=expected)
    print(json.dumps(dict(target=target, verified_files=len(observed))))


def train_argv(base, python, target, mode, run_dir, extra=()):
    return [str(python), '-u', '-m', 'experiments.train_cifar_l8_analog', str(base / 'source' / CONFIG),
            '--mode', mode, '--output-dir', str(run_dir), '--assets', str(base / 'assets/ours'),
            '--device', 'cuda', '--require-gpu-resident', '--target', target,
            '--dataset-root', str((REPO if target == 'local' else REMOTE_RESULTS.parent) / DATASET),
            '--source-identity', str(base / 'source/source_identity.json'), *extra]


def smoke_spec(target):
    """Same trainer, config, assets and runtime as production: four updates, through the queue."""
    from experiments import reporting
    base = REMOTE_RESULTS / STUDY
    run_dir = base / 'smoke/ours'
    return dict(id=JOB.replace('-ours-', '-ours-smoke-'), project_dir=str(REPO), handoff=str(HANDOFF),
        authorization='Execution smoke for exp-034 (four updates, same trainer/config/assets/runtime), riri only.',
        resource_profile='default', peak_memory_mib=11264, headroom_mib=2048, expected_seconds=300,
        budget_seconds=900, stale_seconds=600, priority=7, deadline='2026-10-02T20:00:00+02:00',
        retry_safe=False, recovery=False,
        placements={target: dict(workdir=str(base / 'source'), run_dir=str(run_dir), env=dict(ENV),
            gpu_models=['RTX 5090'], argv=train_argv(base, REMOTE_PYTHON, target, 'smoke', run_dir, ('--smoke-batches', '4')),
            validate_argv=[str(REMOTE_PYTHON), '-m', 'experiments.reporting', 'validate-run', str(run_dir)],
            prepared_files=[dict(path=str(base / name), sha256=reporting.sha256_file(ROOT / name)) for name in
                            ('source/source_identity.json', f'source/{CONFIG}', 'assets/ours/checkpoints/initial_model.pt')])})


def smoke(target):
    from experiments import reporting
    path = ROOT / 'launch/smoke-job.json'
    reporting.atomic_write_json(path, smoke_spec(target))
    subprocess.run([sys.executable, QUEUE, 'submit', str(path)], check=True)


def queue_spec(target):
    from experiments import reporting
    base = REMOTE_RESULTS / STUDY
    run_dir = base / 'cells' / RUN_CELL
    stop = ('import datetime,runpy,sys; seconds=float(sys.argv[1]); '
            'deadline=(datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(seconds=seconds)).isoformat(); '
            "sys.argv=['experiments.train_cifar_l8_analog',*sys.argv[2:],'--stop-before',deadline]; "
            "runpy.run_module('experiments.train_cifar_l8_analog',run_name='__main__')")
    argv = train_argv(base, REMOTE_PYTHON, target, 'train', run_dir,
                      ('--initial-epoch-seconds', '2400', '--deadline-reserve-seconds', '300'))
    argv = [argv[0], '-u', '-c', stop, str(BUDGET_SECONDS - 60), *argv[4:]]
    return dict(id=JOB, project_dir=str(REPO), handoff=str(HANDOFF),
        authorization=('Filip set the goal (October 2) of finding per-block beta values across epochs for the '
                       'digital-ReLU block-output architecture; this BPTT run supplies its trajectory checkpoints. '
                       'Same contract as the Codex linear-output ours arm; runs restricted to riri.'),
        resource_profile='default', peak_memory_mib=11264, headroom_mib=2048, expected_seconds=12000,
        budget_seconds=BUDGET_SECONDS, stale_seconds=2700, startup_grace_seconds=180, priority=3,
        deadline='2026-10-03T08:00:00+02:00', retry_safe=False, recovery=True,
        placements={target: dict(workdir=str(base / 'source'), run_dir=str(run_dir), env=dict(ENV),
            gpu_models=['RTX 5090'], argv=argv,
            validate_argv=[str(REMOTE_PYTHON), '-m', 'experiments.reporting', 'validate-run', str(run_dir)],
            prepared_files=[dict(path=str(base / name), sha256=reporting.sha256_file(ROOT / name)) for name in
                            ('source/source_identity.json', f'source/{CONFIG}', 'assets/ours/checkpoints/initial_model.pt')])},
        completion=dict(argv=[sys.executable, '-m', 'experiments.run_cifar_digital_relu_bptt', 'collect'],
                        workdir=str(REPO), timeout_seconds=1800, retry_safe=True))


def submit(target):
    from experiments import reporting
    path = ROOT / f'launch/{RUN_CELL}-job.json'
    reporting.atomic_write_json(path, queue_spec(target))
    subprocess.run([sys.executable, QUEUE, 'submit', str(path)], check=True)


def validate(run):
    import torch
    from experiments import reporting
    from experiments.analyze_cifar_digital import finite_state
    errors = reporting.validate_run(run)
    assert not errors, errors
    manifest = json.loads((run / 'manifest.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    config = manifest['resolved_config']
    assert config == json.loads((ROOT / 'source' / CONFIG).read_text()), 'Configuration mismatch'
    assert config['block_output_activation'] == 'digital_relu' and config['gradient_method'] == 'bptt'
    assert not manifest['dataset']['official_test_read'] and not result['completion']['official_test_read']
    assert result['completion']['criteria_met'] and result['completion']['full_epoch_budget'], 'Incomplete run'
    rows = list(reporting.iter_metrics(run / 'metrics.jsonl'))
    steps = math.ceil(45000 / config['batch_size'])
    assert [r['epoch'] for r in rows] == list(range(1, config['epochs'] + 1)), 'Incomplete epoch coverage'
    for row in rows:
        assert row['steps'] == row['epoch'] * steps and row['evaluation_split'] == 'validation'
        assert all(math.isfinite(row[k]) for k in ('train_loss', 'validation_loss', 'train_accuracy', 'validation_accuracy'))
    for epoch in (0, *CHECKPOINT_EPOCHS):
        name = 'initial_model' if epoch == 0 else f'epoch_{epoch:03d}'
        saved = torch.load(run / f'checkpoints/{name}.pt', map_location='cpu', weights_only=False)
        assert saved['config'] == config and saved['epoch'] == epoch, name
        assert finite_state(saved['model']), name
        del saved
    return dict(validated=True, epochs=len(rows), terminal_metrics=result['terminal_metrics'],
                validation_accuracy=[r['validation_accuracy'] for r in rows],
                validation_loss=[r['validation_loss'] for r in rows],
                train_accuracy=[r['train_accuracy'] for r in rows], train_loss=[r['train_loss'] for r in rows],
                epoch_seconds=[r['epoch_seconds'] for r in rows] if 'epoch_seconds' in rows[0] else None)


def collect():
    from experiments import reporting
    job = os.environ.get('RUN_WATCH_JOB_ID', JOB)
    state = json.loads(subprocess.check_output([sys.executable, QUEUE, 'status', job], text=True))
    target = state['attempts'][-1]['target']
    out = ROOT / 'cells/ours'
    out.mkdir(parents=True, exist_ok=True)
    remote = state['job']['spec']['placements'][target]['run_dir']
    subprocess.run(['rsync', '-a', '-e', ' '.join(SSH), f'{target}:{remote}/', str(out) + '/'], check=True)
    summary = dict(validate(out), job_id=job, target=target)
    reporting.atomic_write_json(ROOT / 'analysis/ours-collection.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    mode, *args = sys.argv[1:]
    {'stage': stage, 'push': push, 'smoke': smoke, 'submit': submit, 'collect': collect}[mode](*args)
