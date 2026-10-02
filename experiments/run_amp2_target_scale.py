"""Train one frozen rescaled-target B=2 case, then replay its selected checkpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.reporting import (append_metric, atomic_write_json, complete_run,
    fail_run, runtime_context, sha256_file, start_run, update_status_progress, validate_run)

ROOT = Path(__file__).resolve().parents[1]


def completed_training(path, *, smoke=False):
    errors = validate_run(path)
    if errors:
        raise ValueError(errors)
    config = json.loads((path/'config.used.json').read_text())
    epochs = [json.loads(line) for line in (path/'metrics.jsonl').read_text().splitlines()]
    epochs = [r for r in epochs if r.get('kind') == 'epoch']
    expected = 1 if smoke else 30
    if [r['epoch'] for r in epochs] != list(range(1, expected+1)):
        raise ValueError('Missing training epochs')
    metrics = json.loads((path/'metrics.json').read_text())
    if metrics.get('official_test_evaluations', 0) != 0:
        raise ValueError('Official test was accessed')
    if config['training_algorithm'] != 'BP' or config['model_base']['current_amp'] != 2:
        raise ValueError('Wrong scientific case')
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--replay-template', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    template = json.loads(args.replay_template.read_text())
    out = args.output_root.resolve()
    started = time.monotonic()
    manifest = dict(study_id=config['study_id'], run_id=config['arm_id'], arm_id=config['arm_id'],
        evidence_class='ordinary_mnist_selection', dataset='ordinary_mnist_validation',
        configuration=config, command=[sys.executable,*sys.argv], smoke=args.smoke,
        runtime=runtime_context(), official_test_read=False,
        inputs=dict(config_sha256=sha256_file(args.config),replay_template_sha256=sha256_file(args.replay_template)),
        git=dict(commit=os.environ.get('EXPERIMENT_SOURCE_COMMIT'),runner_sha256=sha256_file(Path(__file__))))
    start_run(out,manifest)
    child = None

    def run(argv, stage):
        nonlocal child
        log = out/(stage+'.log')
        with log.open('w') as stream:
            child = subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            last_progress = None
            while child.poll() is None:
                progress = dict(stage=stage,child_pid=child.pid)
                if stage=='training':
                    states = list((out/'training').glob('*/status.json'))
                    if states:
                        status = json.loads(states[0].read_text())
                        progress.update(child_state=status['state'],child_progress=status.get('progress'))
                else:
                    path = out/'replay/status.json'
                    if path.exists():
                        status = json.loads(path.read_text())
                        progress.update(child_state=status['state'],child_progress=status.get('progress'))
                if progress != last_progress:
                    last_progress = progress
                    update_status_progress(out,progress)
                    append_metric(out/'metrics.jsonl',{**progress,'elapsed_seconds':time.monotonic()-started})
                time.sleep(2 if args.smoke else 15)
            if child.returncode:
                raise RuntimeError(f'{stage} exited {child.returncode}: {log}')
            child = None

    try:
        argv = [sys.executable,'-m','experiments.exact_run',str(args.config.resolve()),
                '--output-root',str(out/'training'),'--device','cuda',
                '--checkpoint-every-epoch','--skip-terminal-official-test']
        if args.smoke: argv.append('--smoke')
        run(argv,'training')
        bundles = [p.parent for p in (out/'training').glob('*/manifest.json')]
        if len(bundles)!=1: raise ValueError('Expected one training bundle')
        bundle = bundles[0]
        metrics = completed_training(bundle,smoke=args.smoke)
        arch = config['target_scale_amp2']['architecture']
        depth = int(arch[-1]);iterations={2:6,3:8}[depth]
        case = dict(name=config['arm_id'],architecture=arch,scheme='amp2',
            target_scale=config['target_scale'],checkpoint_epoch=metrics['best_epoch'],
            checkpoint_role='best_validation',bundle=str(bundle),relative_target={2:4.,3:6.}[depth],
            T=iterations,K=iterations,source_hashes={name:sha256_file(bundle/name)
                for name in ('config.used.json','best_model.pt')})
        template.update(cases=[case],study_id=config['study_id'],
                        source_contract={'runtime_source_root':str(ROOT)})
        template['dataset']['root']=config['datasets']['mnist']['params']['root']
        replay_config=out/'replay_config.json'
        atomic_write_json(replay_config,template)
        argv=[sys.executable,'-m','experiments.replay_target_scale_read_noise','--config',str(replay_config),
              '--output-root',str(out/'replay')]
        if args.smoke:argv.append('--smoke')
        run(argv,'replay')
        errors=validate_run(out/'replay')
        if errors:raise ValueError(errors)
        complete_run(out,terminal_metrics=dict(training_epochs=1 if args.smoke else 30,
            best_epoch=metrics['best_epoch'],replay_complete=True,elapsed_seconds=time.monotonic()-started),
            completion=dict(criteria_met=True,coverage_complete=True))
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        fail_run(out,error=error)
        raise


if __name__=='__main__':main()
