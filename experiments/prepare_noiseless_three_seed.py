"""Prepare fixed-beta noiseless seed replicates; no calibration or training."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
import torch
from experiments.prepare_paper_training_completion import build_energy, state_hash

STUDY = Path('results/conv23-noiseless-three-seed-20260927-v1')
CONFIG_ROOT = Path('configs/conv/noiseless_three_seed_20260927')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    STUDY.mkdir(parents=True, exist_ok=True)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    assets, records = [], []
    for depth, target in [(2, 4), (3, 6)]:
        templates = {s: json.loads(Path(f'configs/conv/relative_noise_training_20260925/conv{depth}_{s}_relative{target}_eta1em4_seed0.json').read_text()) for s in ['ours', 'legacy']}
        seed_hashes = []
        for seed in [0, 1, 2]:
            source = Path(f'results/paper-training-completion-20260911-v1/assets/wide_kaiming/conv{depth}/seed{seed}.pt')
            if seed == 0:
                assert sha(source) == templates['ours']['initialization']['checkpoint_sha256']
                assert sha(source) == templates['legacy']['initialization']['checkpoint_sha256']
            draw_hash = None
            for scheme in ['ours', 'legacy']:
                draw_cfg = copy.deepcopy(templates[scheme]); draw_cfg['seed'] = seed
                energy = build_energy(draw_cfg)
                expected = state_hash(energy)
                energy.load(source)
                assert state_hash(energy) == expected, 'Existing asset differs from normal seed draw'
                if draw_hash is None: draw_hash = expected
                assert draw_hash == expected, 'Unmatched scheme initialization'
                assert all(p.state.dtype == torch.float32 and torch.isfinite(p.state).all() for p in energy.params())
                assert all(torch.count_nonzero(p.state) == 0 for p in energy.params() if str(p.name).startswith('Bias'))
            seed_hashes.append(draw_hash)
            checkpoint = STUDY / 'initializers' / f'conv{depth}' / f'seed{seed}.pt'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, checkpoint)
            assets.append(dict(architecture=f'conv{depth}', seed=seed, source=str(source), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), tensor_sha256=draw_hash, reproduced_normal_seed_draw=True, matched_across_schemes=True))
            for scheme, template in templates.items():
                c = copy.deepcopy(template)
                arm = f'conv{depth}_{scheme}_relative{target}_eta0_seed{seed}'
                c.update(study_id=STUDY.name, arm_id=arm, seed=seed, init_checkpoint_path=f'inputs/initializers/conv{depth}_seed{seed}.pt')
                c['initialization'].update(checkpoint_sha256=sha(checkpoint), seed=seed, optimizer_state='fresh_at_initialization', recipe='wide_kaiming')
                c['lab']['epochs'] = 30
                c['datasets']['mnist']['params'].update(shuffle_seed=seed, split_seed=0)
                c['eqprop'].update(endpoint_read_noise_mode='nodewise_relative', endpoint_read_noise_eta=0.0, endpoint_read_noise_std=0.0)
                reference = c.pop('relative_output_calibration')
                c['relative_output_calibration'] = {k: copy.deepcopy(reference[k]) for k in ['target', 'T', 'K', 'base_beta', 'injected_beta', 'definition']}
                c['relative_output_calibration'].update(reference_seed=0, calibrated_for_this_seed=(seed == 0), beta_frozen_across_seeds=True, actual_displacement_for_this_seed=('accepted_seed0_reference' if seed == 0 else 'unmeasured_may_differ'), seed0_reference=reference)
                c['reporting'].update(study_id=STUDY.name, arm_id=arm)
                c['relative_noise_training'].update(eta=0.0, primary_metric='epoch_30_validation_accuracy', calibration_reused=(seed == 0), reference_displacement_seed=0, fixed_seed0_beta=True)
                c['evaluation']['inclusion_rule'] = 'Retain all outcomes; compare final epoch 30 validation accuracy in paired seeds 0,1,2.'
                c['noiseless_replication'] = dict(model_seed=seed, shuffle_seed=seed, split_seed=0, noise_eta=0.0, trained_from_initialization=True, uninterrupted_epochs=30, optimizer_reset_at_epoch10=False, comparison_to_epoch30_noisy_restart='Different optimizer history: noisy continuation resets Adam/RNG after epoch10; these clean controls do not.', beta_policy='Freeze accepted seed0 beta across seeds, identical to fixed learning rates; no seed-specific calibration.', displacement_label=f'seed0-reference D/F={target}')
                dest = CONFIG_ROOT / (arm + '.json')
                dest.write_text(json.dumps(c, indent=2, sort_keys=True) + '\n')
                records.append(dict(arm_id=arm, config=str(dest), config_sha256=sha(dest), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), staged_checkpoint=c['init_checkpoint_path'], base_beta=c['beta'], seed=seed))
        assert len(set(seed_hashes)) == 3, 'Seed draws are not independent'
    report = dict(expected_cases=12, ready_cases=len(records), epochs=30, eta=0, assets=assets, records=records)
    (STUDY/'preparation.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    snapshot = STUDY/'source-training-v1'
    if snapshot.exists():
        for r in records:
            for src, rel in [(Path(r['checkpoint']),Path(r['staged_checkpoint'])),(Path(r['config']),Path(r['config']))]:
                out=snapshot/rel;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out)
    print(json.dumps(dict(ready_cases=len(records), independent_assets=len(assets), staged=snapshot.exists())))


if __name__ == '__main__':
    main()
