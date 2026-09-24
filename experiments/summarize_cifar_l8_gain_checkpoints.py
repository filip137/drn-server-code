"""Read-only final-state checks and BN epsilon shares for the CIFAR gain pair."""
import argparse
import csv
import json
from pathlib import Path

import torch
from experiments import reporting
from experiments.summarize_cifar_l8_bn_ablation import training_rows


def summarize(root):
    root = root.resolve(); out = root / 'analysis'; out.mkdir(exist_ok=True)
    paths = {
        'fixed': root / 'cells/legacy_frozen_bn_fixed_gain',
        'log': root / 'cells/legacy_frozen_bn_log_gain',
        'original_softplus': root.parent / 'cifar10-l8-bn-ablation-seed0-20260923-v1/cells/legacy_frozen_bn',
    }
    cached_path = out / 'gain_checkpoint_summary.json'
    cached = json.loads(cached_path.read_text()) if cached_path.exists() else {}
    cases, rows = {}, []
    for label, run in paths.items():
        checkpoint_path = run / 'checkpoints/final_model.pt'
        if not checkpoint_path.exists():
            continue
        status = json.loads((run / 'status.json').read_text())
        if status['state'] not in ('complete', 'failed'):
            continue
        digest = reporting.sha256_file(checkpoint_path)
        old = cached.get('cases', {}).get(label, {})
        if old.get('checkpoint_sha256') == digest:
            cases[label] = old; rows.extend(old['blocks']); continue
        c = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config = c['config']; epoch = c['epoch']; module = c['model']['module']
        assert config['bn_affine_trainable'] is False
        assert c['scheduler']['last_epoch'] == epoch
        groups = {g['name']: g for g in c['optimizer']['param_groups']}
        steps = {name: [int(c['optimizer']['state'][pid]['step']) for pid in group['params']]
                 for name, group in groups.items()}
        assert all(values == [1407*epoch] for values in steps.values())
        assert not any(name.startswith('bridges.') for name in groups)
        mode = config.get('conv_input_gain_mode', 'softplus')
        assert mode == ('softplus' if label == 'original_softplus' else label)
        assert config['optimizer']['gain_learning_rate'] == 5e-5
        if mode == 'log':
            assert config['optimizer']['conv_gain_learning_rate'] == 1e-3
        measured = training_rows(run, root.parent)
        last = next(row for row in measured if row['epoch'] == epoch)
        epsilon = config.get('batch_norm_eps', [1e-5]*3)
        if not isinstance(epsilon, list):
            epsilon = [epsilon]*3
        blocks = []
        for index in range(3):
            name = f'analog_blocks.{index}._input_gain_raw'
            raw = module[name]
            if mode == 'fixed':
                assert name not in groups and float(raw) == 100.
                gain = float(raw)
            elif mode == 'log':
                gain = float(config['input_gain_init']*raw.exp())
                assert name in groups
            else:
                gain = float(torch.nn.functional.softplus(raw))
                assert name in groups
            # Recorded CUDA value is preferred for reporting float32 exp;
            # CPU and GPU exp may differ by one rounding unit.
            logged_gain = last.get('input_gains', [gain]*3)[index]
            assert abs(logged_gain-gain) <= 2e-5*max(1., abs(gain))
            prefix = f'bridges.{index}.1.'
            assert torch.equal(module[prefix+'weight'], torch.ones_like(module[prefix+'weight']))
            assert torch.equal(module[prefix+'bias'], torch.zeros_like(module[prefix+'bias']))
            tracked = int(module[prefix+'num_batches_tracked'])
            assert tracked == 1407*epoch
            variance = module[prefix+'running_var'].double()
            shares = epsilon[index]/(variance+epsilon[index])
            q = torch.tensor([0., .1, .5, .9, 1.], dtype=torch.float64)
            statistics = last.get('first_batch_boundary_statistics', {})
            pooled = statistics.get(f'block_{index}/pooled_voltage', {})
            boundary = statistics.get(f'block_{index}/boundary_output', {})
            block = {'case': label, 'epoch': epoch, 'block': index+1,
                'gain': logged_gain, 'raw_gain_parameter': float(raw),
                'bn_affine_frozen_verified': True, 'bn_batches_tracked': tracked,
                'bn_epsilon': epsilon[index],
                'running_variance_median': float(variance.median()),
                'epsilon_share_min': float(shares.min()),
                'epsilon_share_p10': float(torch.quantile(shares, q)[1]),
                'epsilon_share_median': float(torch.quantile(shares, q)[2]),
                'epsilon_share_p90': float(torch.quantile(shares, q)[3]),
                'epsilon_share_max': float(shares.max()),
                'first_training_batch_pooled_voltage_rms': pooled.get('rms'),
                'first_training_batch_boundary_output_rms': boundary.get('rms')}
            blocks.append(block); rows.append(block)
        assert 'head._input_gain_raw' in groups
        cases[label] = {'checkpoint': str(checkpoint_path), 'checkpoint_sha256': digest,
            'epoch': epoch, 'mode': mode, 'optimizer_steps': steps,
            'scheduler_last_epoch': c['scheduler']['last_epoch'], 'blocks': blocks,
            'head_gain': float(torch.nn.functional.softplus(module['head._input_gain_raw'])),
            'head_initial_lr': groups['head._input_gain_raw']['initial_lr'],
            'head_next_lr': groups['head._input_gain_raw']['lr'],
            'validation_accuracy': last['validation_accuracy'], 'validation_loss': last['validation_loss']}
        assert cases[label]['head_initial_lr'] == 5e-5
        del c
    summary = {'cases': cases, 'all_three_present': len(cases) == 3,
               'epsilon_share_definition': 'eps/(saved BN running variance+eps), per channel in eval normalization',
               'voltage_measurement': 'tracked pass of first augmented training batch in last epoch; before that epoch training, not final checkpoint replay',
               'checkpoint_read_only': True, 'official_test_read': False}
    reporting.atomic_write_json(cached_path, summary)
    if rows:
        with (out / 'gain_checkpoint_bnstats.csv').open('w') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
