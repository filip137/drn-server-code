"""Prepare approved epoch-10 weights-only restarts; never launch training."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(study: Path, config_root: Path, eta_tags=None) -> dict:
    eta_tags = list(eta_tags) if eta_tags is not None else ['1em4', '3em4', '1em3']
    if not eta_tags or len(set(eta_tags)) != len(eta_tags):
        raise ValueError('eta_tags must be nonempty and unique')
    config_root.mkdir(parents=True, exist_ok=True)
    records, pending = [], []
    old_configs = Path('configs/conv/relative_noise_training_20260925')
    for depth, target in [(2, 4), (3, 6)]:
        for scheme in ['ours', 'legacy']:
            for eta in eta_tags:
                arm = f'conv{depth}_{scheme}_relative{target}_eta{eta}_seed0'
                src = study / 'source_epoch10' / arm
                if not (src / 'result.json').exists():
                    pending.append(arm)
                    continue
                status = json.loads((src / 'status.json').read_text())
                assert status['state'] == 'complete' and status['runtime']['device'] == 'cuda'
                assert status['result_sha256'] == digest(src / 'result.json')
                epochs = [r for r in map(json.loads, (src / 'metrics.jsonl').read_text().splitlines())
                          if r.get('kind') == 'epoch']
                assert [r['epoch'] for r in epochs] == list(range(1, 11))
                manifest = json.loads((src / 'manifest.json').read_text())
                resolved = manifest['configuration']['resolved']
                cfg = json.loads((old_configs / (arm + '.json')).read_text())
                # Only transport paths differ between frozen source and checked-in configs.
                expected = copy.deepcopy(cfg)
                observed = copy.deepcopy(resolved)
                for item in [expected, observed]:
                    item.pop('init_checkpoint_path', None)
                    item['datasets']['mnist']['params'].pop('root', None)
                assert observed == expected, f'source config mismatch: {arm}'
                assert cfg['evaluation']['official_test']['policy'] == 'disabled'
                checkpoint = src / 'final_model.pt'
                checkpoint_hash = digest(checkpoint)
                original_initialization = copy.deepcopy(cfg['initialization'])
                original_checkpoint_path = cfg['init_checkpoint_path']
                cfg['init_checkpoint_path'] = f'inputs/epoch10/{arm}.pt'
                cfg['initialization'].update(checkpoint_sha256=checkpoint_hash,
                    optimizer_state='fresh_user_approved_weights_only_restart',
                    recipe='trained_epoch10_final_weights', source_epoch=10)
                if 'weight_ceiling_sweep' in cfg:
                    cfg['weight_ceiling_sweep']['initializer_checkpoint_sha256'] = checkpoint_hash
                cfg['lab']['epochs'] = 20
                cfg['study_id'] = study.name
                cfg['reporting']['study_id'] = study.name
                cfg['evaluation']['inclusion_rule'] = 'Retain all outcomes; compare continuation epoch 20 (cumulative epoch 30).'
                cfg['relative_noise_training']['primary_metric'] = 'cumulative_epoch_30_validation_accuracy'
                cfg['weights_only_continuation'] = dict(
                    source_study='conv123-relative-noise-training-20260925-v1',
                    source_arm_id=arm, source_epoch=10, additional_epochs=20,
                    cumulative_epoch_offset=10, local_epoch_to_cumulative='local_epoch + 10',
                    source_location=json.loads((src / 'source_location.json').read_text()),
                    source_manifest_sha256=digest(src / 'manifest.json'),
                    source_result_sha256=digest(src / 'result.json'),
                    source_checkpoint_sha256=checkpoint_hash,
                    source_epoch10_validation_accuracy=epochs[-1]['metrics']['validation_accuracy'],
                    original_initialization=original_initialization,
                    original_initializer_path=original_checkpoint_path,
                    optimizer_reset=True, rng_reset=True,
                    reset_policy='Fresh Adam and restart original declared RNG seeds, including read-noise and data ordering; no optimizer/RNG restoration.',
                    user_approved=True, beta_recalibrated=False,
                    displacement_target_refers_to_original_initialization=True)
                dest = config_root / f'{arm}.json'
                dest.write_text(json.dumps(cfg, indent=2, sort_keys=True) + '\n')
                records.append(dict(arm_id=arm, config=str(dest), config_sha256=digest(dest),
                    checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                    staged_checkpoint=cfg['init_checkpoint_path'],
                    source_epoch10_validation_accuracy=epochs[-1]['metrics']['validation_accuracy']))
    report = dict(expected_cases=4 * len(eta_tags), ready_cases=len(records), pending_cases=pending,
                  epochs=20, cumulative_epoch_offset=10, records=records)
    out = study / 'preparation.json'
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, default=Path('results/conv23-relative-noise-epoch30-20260926-v1'))
    parser.add_argument('--config-root', type=Path, default=Path('configs/conv/relative_noise_epoch30_20260926'))
    parser.add_argument('--eta-tags', nargs='+', default=None, help='Filename noise tags, e.g. 1em8 1em7 1em6 1em5')
    args = parser.parse_args()
    report = prepare(args.study_root, args.config_root, args.eta_tags)
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
