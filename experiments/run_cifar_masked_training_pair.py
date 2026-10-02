"""Configure, stage, collect and compare exp-031: float32 hybrid EqProp with/without the force mask.

    python -m experiments.run_cifar_masked_training_pair configure   # targets from exp-030 ours replays
    python -m experiments.run_cifar_masked_training_pair stage       # assets, frozen source, Slurm scripts
    python -m experiments.run_cifar_masked_training_pair collect ARM # rsync from Jean Zay, then validate
    python -m experiments.run_cifar_masked_training_pair compare

Both arms are identical except ``conv_eqprop.force_mask``. The base contract is
exp-026's periodic arm (ours, seed0 initializer, data order, Adam, schedule, T/K,
four calibrations per epoch); precision becomes float32 and the RMS targets and
initial betas come from the frozen exp-030 selection rule.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
STUDY = 'cifar-float32-masked-training-20261001-v1'
ROOT = REPO / 'results' / STUDY
REPLAY = REPO / 'results/cifar-force-mask-float32-replay-20261001-v1'
BASE_CONFIG = REPO / 'configs/cifar/float64_periodic_beta_20261001/periodic.json'
INITIALIZER = REPO / 'results/cifar-layer-rms-all-schemes-float64-20260930-v1/cases/init-ours-e0/inputs/checkpoint.pt'
INITIALIZER_SHA256 = 'c4d98ef636949fed044ce4d2e94544be8c69eb816dc279338d7a067018fa6c37'
CONFIGS = Path('configs/cifar/float32_masked_training_20261001')
ARMS = ('mask', 'nomask')
OURS = ('init-ours-e0', 'eqprop-ours-e1', 'eqprop-ours-e5')
TIE = 1e-3
PYTHON = REPO / 'results/cifar-layer-rms-all-schemes-float64-20260930-v1/runtime/py312/bin/python'
LOCAL_DATA = REPO / 'results/cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1/data/cifar-10-batches-py'
JZ_USER_ROOT = Path('/lustre/fsn1/projects/rech/fmu/ucy17uy/server_code')
JZ_ROOT = JZ_USER_ROOT / 'results' / STUDY
JZ_SOURCE = Path('/lustre/fswork/projects/rech/fmu/ucy17uy/server_code') / f'{STUDY}-source'
JZ_DATA = JZ_USER_ROOT / 'results/cifar10-l8-schedule-renewal-e50-e70-seed0-20260924-v1/inputs/cifar-10-batches-py'
SLURM = dict(account='fmu@v100', partition='gpu_p13', constraint='v100-32g')
PRODUCTION_HOURS = 9
TRAINER_HOURS = 8.5


def load(path):
    return json.loads(Path(path).read_text())


def tied_median(scores, tie=TIE):
    """Index of the median (larger middle) among scores within ``tie`` of the best, in grid order."""
    best = max(scores)
    tied = [i for i, score in enumerate(scores) if score >= best - tie]
    return tied[len(tied) // 2], tied


def select_targets():
    """Frozen exp-030 rule: maximize the worst ours minibatch/layer cosine, then take the tied median."""
    from experiments import reporting
    per_case = {}
    for case in OURS:
        cell = REPLAY / 'cells' / f'{case}-masked'
        receipt = REPLAY / 'analysis' / f'{case}-masked-collection.json'
        assert receipt.is_file(), f'Missing validated exp-030 collection: {receipt}'
        assert load(receipt)['comparison_sha256'] == reporting.sha256_file(cell / 'artifacts/comparison.json')
        per_case[case] = load(cell / 'artifacts/comparison.json')
    grid = sorted({row['target_relative_rms'] for row in per_case[OURS[0]]['layers']})
    selection = []
    for block in (1, 2, 3):
        scores = []
        for index, target in enumerate(grid):
            values = [row['local_bptt']['cosine'] for data in per_case.values() for row in data['batches']
                      if row['block'] == block and row['target_relative_rms'] == target]
            assert values and all(v is not None and math.isfinite(v) for v in values), (block, target)
            scores.append(min(values))
        best = max(scores)
        chosen, tied = tied_median(scores)
        init = load(REPLAY / 'cells/init-ours-e0-masked/artifacts/resolved_beta_vectors.json')
        case_index = init['relative_rms_targets'].index(grid[chosen])
        beta = init['beta_multiplier_vectors'][case_index][block - 1] * init['base_betas'][block - 1]
        selection.append(dict(block=block, target=grid[chosen], score=scores[chosen], best_score=best,
                              tied_targets=[grid[i] for i in tied], initial_beta=beta,
                              scores=dict(zip(map(str, grid), scores))))
    return selection


def make_config(arm, selection):
    config = load(BASE_CONFIG)
    config.update(study_id=STUDY, arm_id=arm, precision='float32', epochs=5,
                  evidence_class='cifar_float32_masked_vs_unmasked_hybrid_training',
                  maximum_runtime_hours=TRAINER_HOURS)
    spec = config['conv_eqprop']
    spec.update(rms_targets=[s['target'] for s in selection], initial_betas=[s['initial_beta'] for s in selection],
                initialization_provenance=f'{REPLAY.name}/cells/init-ours-e0-masked/artifacts/resolved_beta_vectors.json')
    if arm == 'mask':
        spec['force_mask'] = 'inactive_output_channels'
    return config


def configure():
    from experiments import reporting
    selection = select_targets()
    (REPO / CONFIGS).mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        path = REPO / CONFIGS / f'{arm}.json'
        assert not path.exists(), path
        reporting.atomic_write_json(path, make_config(arm, selection))
    reporting.atomic_write_json(ROOT / 'inputs/target_selection.json', dict(rule=(
        'per block: score each sampled target by the minimum local cosine over ours init/e1/e5, layers and '
        f'minibatches of the masked float32 exp-030 replays; take the median (larger middle) of targets within {TIE} '
        'of the best score; initial betas are init-ours-e0 masked calibrated betas at that target'),
        blocks=selection))
    print(json.dumps([{k: s[k] for k in ('block', 'target', 'score', 'best_score', 'tied_targets', 'initial_beta')}
                      for s in selection], indent=1))


def source_files():
    for directory in ('experiments', 'labs', 'model', 'training'):
        for path in sorted((REPO / directory).rglob('*.py')):
            if '__pycache__' not in path.parts and 'tests' not in path.relative_to(REPO).parts:
                yield path.relative_to(REPO)


def sbatch(name, arm, *, qos, hours, smoke):
    run_dir = JZ_ROOT / ('smoke' if smoke else 'cells') / arm
    stop_minutes = int(hours * 60) - 20
    trainer = ['python', '-u', '-m', 'experiments.train_cifar_l8_analog', str(JZ_SOURCE / CONFIGS / f'{arm}.json'),
               '--mode', 'smoke' if smoke else 'train', '--output-dir', str(run_dir), '--device', 'cuda',
               '--require-gpu-resident', '--target', 'jean-zay', '--assets', str(JZ_ROOT / 'assets'),
               '--dataset-root', str(JZ_DATA), '--source-identity', str(JZ_SOURCE / 'source_identity.json'),
               '--initial-epoch-seconds', '600' if smoke else '4800', '--deadline-reserve-seconds', '300']
    if smoke:
        trainer += ['--smoke-batches', '4']
    lines = ['#!/bin/bash -l', f'#SBATCH --job-name={name}', f"#SBATCH --account={SLURM['account']}",
             f"#SBATCH --partition={SLURM['partition']}", f'#SBATCH --qos={qos}',
             f"#SBATCH --constraint={SLURM['constraint']}", '#SBATCH --gres=gpu:1', '#SBATCH --nodes=1',
             '#SBATCH --ntasks=1', '#SBATCH --cpus-per-task=10', '#SBATCH --hint=nomultithread',
             f'#SBATCH --time={int(hours):02d}:{int(round((hours % 1) * 60)):02d}:00',
             f'#SBATCH --output={JZ_ROOT}/slurm/{name}-%j.out', f'#SBATCH --error={JZ_ROOT}/slurm/{name}-%j.err',
             'set -euo pipefail', 'module purge', 'module load pytorch-gpu/py3/2.5.0',
             'export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1',
             'export KMP_DISABLE_SHM=1 KMP_SHM_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1',
             'export CUBLAS_WORKSPACE_CONFIG=:4096:8', f'export PYTHONPATH={JZ_SOURCE}', 'unset LD_PRELOAD',
             f'cd {JZ_SOURCE}', f'STOP=$(date -u -d "+{stop_minutes} minutes" +%Y-%m-%dT%H:%M:%S+00:00)',
             'srun ' + ' '.join(trainer) + ' --stop-before "$STOP"', '']
    return '\n'.join(lines)


def stage():
    from experiments import reporting
    assert not (ROOT / 'source').exists(), 'Frozen source already staged'
    assert reporting.sha256_file(INITIALIZER) == INITIALIZER_SHA256
    source = ROOT / 'source'
    files = [*source_files(), *sorted(p.relative_to(REPO) for p in (REPO / CONFIGS).glob('*.json'))]
    for relative in files:
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, source / relative)
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    reporting.atomic_write_json(source / 'source_identity.json', dict(commit=commit, dirty=True, study_id=STUDY,
        note='exp-031 float32 hybrid EqProp, mask versus no mask; uncommitted files identified by hash.',
        source_sha256={str(r): reporting.sha256_file(source / r) for r in files}))
    shutil.copy2(INITIALIZER, ROOT / 'inputs/initial_model.pt')
    reporting.atomic_write_json(ROOT / 'inputs/provenance.json', dict(initial_checkpoint_source=str(
        INITIALIZER.relative_to(REPO)), initial_sha256=INITIALIZER_SHA256))
    subprocess.run([str(PYTHON), '-m', 'experiments.train_cifar_l8_analog', str(source / CONFIGS / 'mask.json'),
                    '--mode', 'prepare', '--output-dir', str(ROOT / 'assets'), '--device', 'cpu',
                    '--dataset-root', str(LOCAL_DATA), '--source-identity', str(source / 'source_identity.json'),
                    '--initial-checkpoint', str(ROOT / 'inputs/initial_model.pt'), '--target', 'local-preparation'],
                   cwd=source, check=True, env=dict(__import__('os').environ, KMP_DISABLE_SHM='1', OMP_NUM_THREADS='1',
                                                     CUDA_VISIBLE_DEVICES=''))
    (ROOT / 'launch').mkdir(parents=True, exist_ok=True)
    (ROOT / 'launch/canary-mask.sbatch').write_text(sbatch('cifar-fp32-mask-canary', 'mask', qos='qos_gpu-dev',
                                                           hours=1, smoke=True))
    for arm in ARMS:
        (ROOT / f'launch/{arm}.sbatch').write_text(sbatch(f'cifar-fp32-{arm}', arm, qos='qos_gpu-t3',
                                                          hours=PRODUCTION_HOURS, smoke=False))
    print(json.dumps(dict(source_files=len(files), assets=str(ROOT / 'assets')), indent=1))


def validate(arm, run_dir=None):
    import torch
    from experiments import reporting
    out = Path(run_dir) if run_dir else ROOT / 'cells' / arm
    errors = reporting.validate_run(out)
    assert not errors, errors
    manifest = load(out / 'manifest.json')
    config = manifest['resolved_config']
    assert config == load(ROOT / 'source' / CONFIGS / f'{arm}.json')
    assert config['precision'] == 'float32' and config['epochs'] == 5
    assert config['conv_eqprop'].get('force_mask', 'none') == ('inactive_output_channels' if arm == 'mask' else 'none')
    rows = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    rows = [r for r in rows if 'validation_accuracy' in r]
    assert [r['epoch'] for r in rows] == list(range(1, 6))
    assert all(r['steps'] == r['epoch'] * 1407 and r['train_examples'] == 45000 and r['validation_examples'] == 5000
               for r in rows)
    assert all(('force_mask_statistics' in r) == (arm == 'mask') for r in rows)
    final = torch.load(out / 'checkpoints/final_model.pt', map_location='cpu', weights_only=False)
    assert final['epoch'] == final['scheduler']['last_epoch'] == 5
    assert {int(v['step']) for v in final['optimizer']['state'].values()} == {7035}
    calibrations = list((out / 'artifacts').glob('beta_calibration_*.json'))
    assert len(calibrations) == 20 and all(load(p)['complete'] for p in calibrations)
    result = load(out / 'result.json')
    assert result['completion']['full_epoch_budget'] and not result['completion']['official_test_read']
    return dict(arm=arm, validated=True, epochs=5, steps=7035, calibrations=len(calibrations),
                validation_accuracy=[r['validation_accuracy'] for r in rows],
                validation_loss=[r['validation_loss'] for r in rows],
                force_mask_statistics=[r.get('force_mask_statistics') for r in rows],
                betas=[r['beta_state']['values'] for r in rows])


def collect(arm):
    """Queue completion hook: copy the cell from its actual lab placement, then validate."""
    import os
    from experiments import reporting
    queue = '/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py'
    job = os.environ.get('RUN_WATCH_JOB_ID', f'cifar-fp32-masked-training-{arm}-20261001')
    state = json.loads(subprocess.check_output([sys.executable, queue, 'status', job], text=True))
    target = state['attempts'][-1]['target']
    assert target in LAB_PLACEMENTS, target
    out = ROOT / 'cells' / arm
    if target != 'local':
        out.mkdir(parents=True, exist_ok=True)
        remote = f"{target}:{state['job']['spec']['placements'][target]['run_dir']}/"
        subprocess.run(['rsync', '-a', '-e', 'ssh -F /home/filip/.ssh/config -o BatchMode=yes', remote, str(out) + '/'],
                       check=True)
    summary = dict(validate(arm), job_id=job, target=target)
    reporting.atomic_write_json(ROOT / 'analysis' / f'{arm}-collection.json', summary)
    print(json.dumps(summary), flush=True)


LAB_PLACEMENTS = ('local', 'nom-cool-1')
LAB_BUDGET_SECONDS = 36000


def lab_spec(arm):
    """Shared-queue job: both arms on RTX 3090s with the same pinned torch2.11 runtime."""
    from experiments import reporting
    env = dict(KMP_DISABLE_SHM='1', KMP_SHM_DISABLE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8')
    placements = {}
    for target in LAB_PLACEMENTS:
        local = target == 'local'
        base = ROOT if local else Path('/home/filip/server_code/results') / STUDY
        python = PYTHON if local else Path('/home/filip/server_code/results/cifar-layer-rms-all-schemes-float64-20260930-v1/runtime/py312/bin/python')
        data = LOCAL_DATA if local else Path('/home/filip/server_code/results/cifar10-l8-analog-three-scheme-adam-lr-seed0-20260921-v1/data/cifar-10-batches-py')
        run_dir = base / 'cells' / arm
        stop_minutes = LAB_BUDGET_SECONDS // 60 - 30
        command = ' '.join([str(python), '-u', '-m', 'experiments.train_cifar_l8_analog',
            str(base / 'source' / CONFIGS / f'{arm}.json'), '--mode', 'train', '--output-dir', str(run_dir),
            '--device', 'cuda', '--require-gpu-resident', '--target', target, '--assets', str(base / 'assets'),
            '--dataset-root', str(data), '--source-identity', str(base / 'source/source_identity.json'),
            '--initial-epoch-seconds', '4800', '--deadline-reserve-seconds', '300',
            '--stop-before', f'"$(date -u -d "+{stop_minutes} minutes" +%Y-%m-%dT%H:%M:%S+00:00)"'])
        placements[target] = dict(workdir=str(base / 'source'), run_dir=str(run_dir), env=env,
            gpu_models=['RTX 3090'], argv=['bash', '-c', 'exec ' + command],
            validate_argv=[str(python), '-m', 'experiments.reporting', 'validate-run', str(run_dir)],
            prepared_files=[dict(path=str(base / 'source/source_identity.json'),
                                 sha256=reporting.sha256_file(ROOT / 'source/source_identity.json')),
                            dict(path=str(base / 'source' / CONFIGS / f'{arm}.json'),
                                 sha256=reporting.sha256_file(ROOT / 'source' / CONFIGS / f'{arm}.json')),
                            dict(path=str(base / 'assets/checkpoints/initial_model.pt'),
                                 sha256=reporting.sha256_file(ROOT / 'assets/checkpoints/initial_model.pt'))])
    return dict(id=f'cifar-fp32-masked-training-{arm}-20261001', project_dir=str(REPO),
        handoff=str(REPO / 'campaigns/cifar10-beta-selection/series/001-beta-selection/experiments/exp-031-float32-masked-training.md'),
        authorization=('Filip asked for a short float32 training comparison after the masked replays: two fresh '
                       'five-epoch ours hybrid EqProp arms differing only in the inactive-channel force mask.'),
        resource_profile='default', peak_memory_mib=11264, headroom_mib=2048, priority=10,
        expected_seconds=21600, budget_seconds=LAB_BUDGET_SECONDS, deadline='2026-10-02T12:00:00+02:00',
        retry_safe=False, recovery=False, stale_seconds=2700, placements=placements,
        completion=dict(argv=[sys.executable, '-m', 'experiments.run_cifar_masked_training_pair', 'collect', arm],
                        workdir=str(REPO), timeout_seconds=1800, retry_safe=True))


def write_lab_specs():
    from experiments import reporting
    for arm in ARMS:
        reporting.atomic_write_json(ROOT / 'launch' / f'{arm}-lab.json', lab_spec(arm))
    print('written')


def compare():
    print('comparison is written after both arms are collected; see exp-031 handoff', flush=True)


if __name__ == '__main__':
    mode, *args = sys.argv[1:]
    {'configure': configure, 'stage': stage, 'collect': collect, 'validate': validate, 'compare': compare,
     'lab-specs': write_lab_specs}[mode](*args)
