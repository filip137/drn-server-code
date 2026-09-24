"""Read-only gain extraction from the twelve locally collected checkpoints."""
import csv
import argparse
import hashlib
import json
from pathlib import Path

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('root', type=Path, help='Collected CIFAR mechanism study root')
root = parser.parse_args().root.resolve()
rows = []
affine_rows = []
for scheme in ('baseline', 'ours', 'legacy'):
    for epoch in (0, 10, 30, 50):
        path = root / 'inputs' / f'{scheme}_e{epoch}.pt'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        groups = {g['name']: g for g in checkpoint['optimizer']['param_groups']}
        for block in range(3):
            gamma = checkpoint['model']['module'][f'bridges.{block}.1.weight'].double()
            beta = checkpoint['model']['module'][f'bridges.{block}.1.bias'].double()
            quantiles = torch.quantile(gamma, torch.tensor([0., .1, .5, .9, 1.], dtype=torch.float64))
            affine_rows.append(dict(
                scheme=scheme, epoch=epoch, block=block+1, channels=gamma.numel(),
                gamma_quantiles=dict(zip(['min', 'p10', 'median', 'p90', 'max'], quantiles.tolist())),
                gamma_median_absolute_change=torch.quantile((gamma-1).abs(), .5).item(),
                beta_rms=beta.square().mean().sqrt().item(), checkpoint_sha256=digest))
        for name, raw in checkpoint['model']['module'].items():
            if '_input_gain_raw' not in name:
                continue
            gain = torch.nn.functional.softplus(raw).item()
            rows.append(dict(scheme=scheme, epoch=epoch, parameter=name,
                             raw=raw.item(), gain=gain,
                             change_from_100_percent=gain - 100.,
                             next_lr=groups[name]['lr'],
                             checkpoint=str(path.relative_to(root)), sha256=digest))
with (root / 'analysis/input_gains.csv').open('w') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

with (root / 'analysis/gradients_adam.csv').open() as handle:
    updates = [row for row in csv.DictReader(handle)
               if row['epoch'] == '50' and '_input_gain_raw' in row['parameter']
               and row['schedule'] == 'operational' and row['mode'] == 'training']
summary = {}
for scheme in ('baseline', 'ours', 'legacy'):
    selected = [r for r in updates if r['source'] == f'{scheme}_e50'
                and r['parameter'].startswith('analog_blocks.')]
    summary[scheme] = dict(
        block_batch_measurements=len(selected),
        zero_float32_updates=sum(float(r['projected_adam_proposal_l2']) == 0
                                 for r in selected),
        maximum_unrounded_update=max(float(r['adam_proposal_l2']) for r in selected))
ulp = (torch.nextafter(torch.tensor(100.), torch.tensor(float('inf'))) - 100.).item()
(root / 'analysis/input_gain_summary.json').write_text(json.dumps(
    dict(float32_spacing_at_100=ulp, epoch50_next_update_replay=summary,
         final_values=[r for r in rows if r['epoch'] == 50]), indent=2) + '\n')
print(json.dumps(dict(float32_spacing_at_100=ulp, epoch50_next_update_replay=summary), indent=2))
(root / 'analysis/bn_affine_checkpoint_summary.json').write_text(
    json.dumps(affine_rows, indent=2) + '\n')

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, scheme in zip(axes, ('baseline', 'ours', 'legacy')):
    for block in range(4):
        parameter = f'analog_blocks.{block}._input_gain_raw' if block < 3 else 'head._input_gain_raw'
        selected = [r for r in rows if r['scheme'] == scheme and r['parameter'] == parameter]
        ax.plot([r['epoch'] for r in selected], [r['change_from_100_percent'] for r in selected],
                marker='o', label=f'Block {block+1}' if block < 3 else 'Classifier')
    ax.axhline(0, color='grey', linewidth=.6)
    ax.set_title(scheme); ax.set_xlabel('Saved epoch'); ax.set_xticks([0, 10, 30, 50])
    ax.grid(alpha=.2)
axes[0].set_ylabel('Gain change from initialization (%)')
axes[-1].legend(fontsize=8)
fig.suptitle('Learned scalar gains: initialized at 100; saved checkpoints only')
fig.tight_layout()
fig.savefig(root / 'analysis/input_gain_trajectories.png', dpi=160)
plt.close(fig)
