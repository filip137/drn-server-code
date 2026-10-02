"""Train the two approved heads and replay saved epochs with existing runners."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments import reporting

SOURCE = Path(__file__).resolve().parents[1]
ARMS = ('single', 'hidden512')
EPOCHS = (1, 5)


def read(path):
    return json.loads(path.read_text())


def validate_training(root, arm):
    import torch
    from experiments.replay_cifar_nonlinear_head import snapshot_hashes
    run = root / 'train' / arm
    errors = reporting.validate_run(run)
    if errors:
        raise ValueError(errors)
    if read(run / 'status.json')['state'] != 'complete':
        raise ValueError('Training has not completed')
    config = read(run / 'manifest.json')['resolved_config']
    if config != read(root / 'inputs' / f'{arm}.json'):
        raise ValueError('Training config changed')
    if config['gradient_method'] != 'bptt' or config['precision'] != 'float32' or config['epochs'] != 5:
        raise ValueError('Wrong training contract')
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    rows = [row for row in rows if 'validation_accuracy' in row]
    if [row['epoch'] for row in rows] != list(range(1, 6)):
        raise ValueError('Incomplete five-epoch coverage')
    if any(row['train_examples'] != 45000 or row['validation_examples'] != 5000 or
           row['steps'] != row['epoch'] * 1407 for row in rows):
        raise ValueError('Incomplete epoch or validation split')
    initial = torch.load(run / 'checkpoints/initial_model.pt', map_location='cpu', weights_only=False)
    source = torch.load(root / 'inputs' / f'{arm}.pt', map_location='cpu', weights_only=False)
    if snapshot_hashes(initial['model']) != snapshot_hashes(source['model']):
        raise ValueError('Training did not start from the declared initializer')
    for epoch in EPOCHS:
        checkpoint = torch.load(run / f'checkpoints/epoch_{epoch:03d}.pt', map_location='cpu', weights_only=False)
        if checkpoint['epoch'] != epoch or checkpoint['scheduler']['last_epoch'] != epoch:
            raise ValueError('Saved epoch boundary differs')
        if checkpoint['config'] != config:
            raise ValueError('Checkpoint config differs')
        if {int(s['step']) for s in checkpoint['optimizer']['state'].values()} != {epoch * 1407}:
            raise ValueError('Optimizer steps differ')
        if any(t.dtype != torch.float32 for group in checkpoint['model'].values()
               for t in group.values() if t.is_floating_point()):
            raise ValueError('Training precision differs')
    result = read(run / 'result.json')
    if result['terminal_metrics']['final_solver_audit_passed'] is not None:
        raise ValueError('Unexpected solver audit')
    return dict(epochs=5, steps=7035, metrics=rows,
                initial_trunk_sha256=snapshot_hashes(initial['model'])['trunk_sha256'])


def validate(root, arm, require_outer=True):
    from experiments.replay_cifar_nonlinear_head import validate_run
    training = validate_training(root, arm)
    diagnostics = {str(epoch): validate_run(root / 'replay' / f'e{epoch}' / arm) for epoch in EPOCHS}
    training_manifest = read(root / 'train' / arm / 'manifest.json')
    for epoch in EPOCHS:
        config = read(root / 'replay' / f'e{epoch}' / arm / 'manifest.json')['resolved_config']
        checkpoint = root / 'train' / arm / f'checkpoints/epoch_{epoch:03d}.pt'
        provenance = config['training_provenance']
        digest = reporting.sha256_file(checkpoint)
        if (config['checkpoint_epoch'] != epoch or config['checkpoint_sha256'] != digest or
                provenance['training_checkpoint_sha256'] != digest or
                config['model_config'] != training_manifest['resolved_config'] or
                provenance['training_config'] != training_manifest['resolved_config'] or
                provenance['training_source_sha256'] != training_manifest['source_identity']['source_sha256'] or
                provenance['initial_trunk_sha256'] != training['initial_trunk_sha256']):
            raise ValueError('Replay is not bound to the collected training checkpoint')
    if require_outer:
        errors = reporting.validate_run(root / 'cells' / arm)
        if errors:
            raise ValueError(errors)
        if (read(root / 'cells' / arm / 'status.json')['state'] != 'complete' or
                not read(root / 'cells' / arm / 'result.json')['completion']['criteria_met']):
            raise ValueError('Pipeline has not completed')
    return dict(arm=arm, training=training, diagnostics=diagnostics)


def run_child(command, child_dir, outer, stage, deadline):
    reporting.update_status_progress(outer, dict(stage=stage, child_state='starting'))
    process = subprocess.Popen(command, start_new_session=True)
    previous = None
    try:
        while True:
            try:
                code = process.wait(timeout=min(10, max(.01, deadline-time.monotonic())))
                if code:
                    raise subprocess.CalledProcessError(code, command)
                if read(child_dir / 'status.json')['state'] != 'complete':
                    raise RuntimeError(f'{stage} exited without complete evidence')
                return
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Declared pipeline budget exhausted')
                status = child_dir / 'status.json'
                if status.exists():
                    state = read(status)
                    current = dict(stage=stage, child_state=state['state'], child_progress=state.get('progress', {}))
                    if current != previous:
                        reporting.update_status_progress(outer, current)
                        previous = current
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def run(args):
    import torch
    from experiments.replay_cifar_nonlinear_head import snapshot_hashes
    root, arm = args.root, args.arm
    outer = root / 'cells' / arm
    config_path = root / 'inputs' / f'{arm}.json'
    config = read(config_path)
    identity = read(SOURCE / 'source_identity.json')
    if str(torch.__version__) != '2.11.0+cu128' or torch.version.cuda != '12.8':
        raise ValueError('Training runtime differs from matched replay runtime')
    start = time.monotonic()
    deadline = start + args.budget_seconds
    reporting.start_run(outer, dict(study_id=root.name, run_id=arm, arm_id=arm, seed=0, smoke=False,
        evidence_class='five_epoch_bptt_then_checkpoint_replay', resolved_config=config,
        source_identity=identity, command=sys.argv, target=args.target,
        runtime=reporting.runtime_context(), budget_seconds=args.budget_seconds,
        official_test_read=False))
    def interrupted(signum, _frame):
        # Raise through run_child's finally so its isolated process group stops.
        raise InterruptedError(f'Pipeline received signal {signum}')
    handlers = {signum: signal.signal(signum, interrupted) for signum in (signal.SIGTERM, signal.SIGHUP)}
    try:
        common = [sys.executable, '-u', '-m', 'experiments.train_cifar_l8_analog', str(config_path),
                  '--dataset-root', str(args.dataset_root), '--source-identity', str(SOURCE / 'source_identity.json'),
                  '--target', args.target]
        assets = root / 'assets' / arm
        run_child(common + ['--mode', 'prepare', '--device', 'cpu', '--initial-checkpoint',
                  str(root / 'inputs' / f'{arm}.pt'), '--output-dir', str(assets)], assets, outer, 'prepare', deadline)
        training = root / 'train' / arm
        stop_before = datetime.fromtimestamp(time.time()+7200, timezone.utc).isoformat()
        run_child(common + ['--mode', 'train', '--device', 'cuda', '--require-gpu-resident',
                  '--assets', str(assets), '--output-dir', str(training), '--stop-before', stop_before,
                  '--initial-epoch-seconds', '900'], training, outer, 'training', deadline)
        training_summary = validate_training(root, arm)
        for epoch in EPOCHS:
            checkpoint = training / f'checkpoints/epoch_{epoch:03d}.pt'
            replay_config = dict(read(root / 'inputs/diagnostic.json'), study_id=root.name, arm_id=arm,
                model_config=config, checkpoint_epoch=epoch, checkpoint_sha256=reporting.sha256_file(checkpoint),
                source_sha256=identity['source_sha256'], training_provenance=dict(
                    initial_trunk_sha256=training_summary['initial_trunk_sha256'], training_run=str(training),
                    training_checkpoint_sha256=reporting.sha256_file(checkpoint), training_config=config,
                    training_source_sha256=identity['source_sha256']))
            config_file = outer / f'artifacts/replay-e{epoch}.json'
            reporting.atomic_write_json(config_file, replay_config)
            replay = root / 'replay' / f'e{epoch}' / arm
            remaining = deadline-time.monotonic()-30
            if remaining < 60:
                raise TimeoutError('No remaining replay allowance')
            run_child([sys.executable, '-u', '-m', 'experiments.replay_cifar_nonlinear_head',
                '--config', str(config_file), '--checkpoint', str(checkpoint), '--dataset-root', str(args.dataset_root),
                '--run-dir', str(replay), '--target', args.target, '--budget-seconds', str(min(1740, remaining))],
                replay, outer, f'replay_epoch_{epoch}', deadline)
        evidence = validate(root, arm, require_outer=False)
        reporting.atomic_write_json(outer / 'artifacts/validation.json', evidence)
        reporting.complete_run(outer, terminal_metrics=dict(elapsed_seconds=time.monotonic()-start,
            final_validation_accuracy=training_summary['metrics'][-1]['validation_accuracy']),
            completion=dict(criteria_met=True, training_epochs=5, replay_epochs=list(EPOCHS), official_test_read=False))
    except BaseException as error:
        reporting.fail_run(outer, error=error)
        raise
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def collect(root, arm):
    queue = '/home/filip/.codex/skills/run-watch/scripts/gpu_queue.py'
    status = json.loads(subprocess.check_output([sys.executable, queue, 'status', os.environ['RUN_WATCH_JOB_ID']], text=True))
    target = status['attempts'][-1]['target']
    if target not in ('fifi', 'riri'):
        raise ValueError('Unexpected collection host')
    remote = f'filip@{target}:/home/filip/server_code/results/{root.name}'
    for rel in [f'cells/{arm}', f'assets/{arm}', f'train/{arm}', *(f'replay/e{e}/{arm}' for e in EPOCHS)]:
        dest = root / rel
        dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync', '-a', '-e', 'ssh -F /home/filip/.ssh/config -o BatchMode=yes',
                        remote+'/'+rel+'/', str(dest)+'/'], check=True)
    evidence = validate(root, arm)
    analysis = root / 'analysis'
    analysis.mkdir(exist_ok=True)
    reporting.atomic_write_json(analysis / f'{arm}-collection.json', dict(job_id=os.environ['RUN_WATCH_JOB_ID'],
                                                                       target=target, **evidence))
    with (analysis / '.pair-analysis.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not all((analysis / f'{a}-collection.json').exists() for a in ARMS):
            return
        from experiments.analyze_cifar_nonlinear_head import analyze
        comparisons = {str(e): analyze(root / 'replay' / f'e{e}') for e in EPOCHS}
        summaries = {a: read(analysis / f'{a}-collection.json') for a in ARMS}
        reporting.atomic_write_json(analysis / 'comparison.json', dict(complete=True,
            training={a:s['training'] for a,s in summaries.items()}, epochs=comparisons,
            interpretation_pending=True, official_test_read=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['run', 'validate', 'collect'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--target', default='local')
    parser.add_argument('--budget-seconds', type=float, default=10740)
    args = parser.parse_args()
    args.root = args.root.resolve()
    if args.mode == 'run':
        if args.dataset_root is None:
            parser.error('--dataset-root required for run')
        run(args)
    elif args.mode == 'validate':
        print(json.dumps(validate(args.root, args.arm), indent=2))
    else:
        collect(args.root, args.arm)


if __name__ == '__main__':
    main()
