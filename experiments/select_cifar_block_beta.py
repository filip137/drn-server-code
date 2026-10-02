"""Apply the frozen H-029 rule: per-block output-RMS targets (and a fixed-beta test) across checkpoints.

    python -m experiments.select_cifar_block_beta STUDY_ROOT [--group relu] [--out NAME]

Reads ``cases.json`` and each collected ``cells/<case>/artifacts/comparison.json``.
Strict pass: every layer's pooled local cosine >= 0.99, every minibatch >= 0.95 and
pooled norm ratio within 0.9-1.1. Writes ``analysis/<NAME>.json`` and ``analysis/<NAME>.jpg``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def strict_table(data, block):
    grid = sorted({r['target_relative_rms'] for r in data['layers']})
    out = {}
    for t in grid:
        pooled = [r for r in data['layers'] if r['block'] == block and r['target_relative_rms'] == t]
        batches = [r for r in data['batches'] if r['block'] == block and r['target_relative_rms'] == t]
        ok = (min(r['local_bptt']['cosine'] for r in pooled) >= .99
              and min(r['local_bptt']['cosine'] for r in batches) >= .95
              and all(.9 <= r['local_bptt']['norm_ratio'] <= 1.1 for r in pooled))
        out[t] = dict(passes=ok, beta=pooled[0]['beta_injected_B'],
                      worst_pooled=min(r['local_bptt']['cosine'] for r in pooled),
                      worst_batch=min(r['local_bptt']['cosine'] for r in batches))
    return grid, out


def runs(grid, passing):
    """Contiguous runs (lists of targets) of the passing set, in grid order."""
    result, current = [], []
    for t in grid:
        if t in passing:
            current.append(t)
        elif current:
            result.append(current); current = []
    if current:
        result.append(current)
    return result


def select(root, group=None):
    cases = [r for r in json.loads((root / 'cases.json').read_text())
             if (group is None or r['group'] == group) and (root / 'analysis' / f"{r['id']}-collection.json").is_file()]
    cases.sort(key=lambda r: r['epoch'])
    tables = {r['id']: json.loads((root / 'cells' / r['id'] / 'artifacts/comparison.json').read_text()) for r in cases}
    blocks = {}
    for block in (1, 2, 3):
        per_case, grid = {}, None
        for r in cases:
            grid, table = strict_table(tables[r['id']], block)
            per_case[r['id']] = table
        passing = [t for t in grid if all(per_case[c][t]['passes'] for c in per_case)]
        contiguous = runs(grid, set(passing))
        longest = max(contiguous, key=len) if contiguous else []
        recommended = None
        if longest:
            midpoint = math.exp((math.log(longest[0]) + math.log(longest[-1])) / 2)
            recommended = min(longest, key=lambda t: abs(math.log(t) - math.log(midpoint)))
        beta_ranges = {}
        for c, table in per_case.items():
            own = [t for t in grid if table[t]['passes']]
            own_run = max(runs(grid, set(own)), key=len) if own else []
            beta_ranges[c] = [table[own_run[0]]['beta'], table[own_run[-1]]['beta']] if own_run else None
        valid = [v for v in beta_ranges.values() if v]
        common_beta = ([max(v[0] for v in valid), min(v[1] for v in valid)]
                       if len(valid) == len(beta_ranges) and max(v[0] for v in valid) <= min(v[1] for v in valid) else None)
        blocks[block] = dict(
            passing_targets=passing, contiguous=len(contiguous) <= 1, longest_run=longest,
            recommended_target=recommended,
            margin_steps=(None if recommended is None else
                          [longest.index(recommended), len(longest) - 1 - longest.index(recommended)]),
            beta_at_recommended={c: per_case[c][recommended]['beta'] for c in per_case} if recommended else None,
            worst_at_recommended={c: [per_case[c][recommended]['worst_pooled'], per_case[c][recommended]['worst_batch']]
                                  for c in per_case} if recommended else None,
            per_case_beta_window=beta_ranges, common_fixed_beta_window=common_beta,
            beta_span_at_recommended=(max(per_case[c][recommended]['beta'] for c in per_case) /
                                      min(per_case[c][recommended]['beta'] for c in per_case)) if recommended else None)
    verdict = ('supports' if all(len(b['passing_targets']) >= 2 for b in blocks.values()) else
               'contradicts' if any(not b['passing_targets'] for b in blocks.values()) else 'partial')
    return dict(cases=[dict(id=r['id'], epoch=r['epoch']) for r in cases], blocks=blocks, h029_rule=verdict)


def plot(summary, root, name, tables_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for block, ax in zip((1, 2, 3), axes):
        info = summary['blocks'][block]
        epochs = [c['epoch'] for c in summary['cases']]
        windows = info['per_case_beta_window']
        for case, epoch in zip([c['id'] for c in summary['cases']], epochs):
            window = windows[case]
            if window:
                ax.plot([epoch, epoch], window, color='tab:gray', lw=5, alpha=.35, solid_capstyle='butt')
        if info['beta_at_recommended']:
            ax.plot(epochs, [info['beta_at_recommended'][c['id']] for c in summary['cases']], 'o-', color='tab:orange',
                    label=f"beta at target RMS {info['recommended_target']:g}")
        ax.set(yscale='log', xlabel='Checkpoint epoch', title=f'block {block}')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Injected beta (grey: strict passing range)')
    fig.suptitle('Per-block beta across training: strict passing range and beta at the selected output-RMS target')
    fig.tight_layout()
    fig.savefig(root / 'analysis' / f'{name}.jpg', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--group')
    parser.add_argument('--out', default='beta_selection')
    args = parser.parse_args()
    from experiments import reporting
    summary = select(args.root, args.group)
    reporting.atomic_write_json(args.root / 'analysis' / f'{args.out}.json', summary)
    plot(summary, args.root, args.out, args.root)
    print(json.dumps({b: {k: v for k, v in info.items() if k in ('passing_targets', 'recommended_target', 'margin_steps',
                                                                  'common_fixed_beta_window', 'beta_span_at_recommended')}
                      for b, info in summary['blocks'].items()} | {'rule': summary['h029_rule']}, indent=1))
