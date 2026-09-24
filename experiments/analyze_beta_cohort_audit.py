"""Collect the completed baseline beta audit, retaining each layer/batch failure."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.run_beta_cohort_audit import OUT, ROOT, validated_result


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flag(value):
    return str(value).lower() == 'true'


def main():
    execution = json.loads((OUT / 'execution.json').read_text())
    if execution['state'] != 'complete':
        raise ValueError('Wait for terminal selection and conditional confirmation coverage.')
    cases = execution['runs']
    if sum(r['phase'] == 'selection' for r in cases) != 9:
        raise ValueError('Missing selection cases.')
    expected_confirmation = 3 * sum(d['selected_beta'] is not None for d in execution['decisions'].values())
    if sum(r['phase'] == 'confirmation' for r in cases) != expected_confirmation:
        raise ValueError('Missing conditional confirmation cases.')
    analysis = OUT / 'analysis'
    analysis.mkdir(exist_ok=True)
    all_rows = []; case_rows = []; source_hashes = {}
    for case in cases:
        checked = validated_result(case)
        if checked['result_sha256'] != case['result_sha256']:
            raise ValueError('A collected result changed.')
        directory = (ROOT / case['result']).parent
        source_hashes[case['result']] = case['result_sha256']
        raw = list(csv.DictReader((directory / 'layer_metrics.csv').open()))
        for row in raw:
            row.update(phase=case['phase'], model_seed=case['seed'],
                       cohort='historical' if int(row['batch_index']) < 4 else case['phase'])
            all_rows.append(row)
        initial = [r for r in raw if r['checkpoint_role'] == 'reconstructed_initialization']
        trained = [r for r in raw if r['checkpoint_role'] == 'best_validation']
        case_rows.append(dict(architecture=case['architecture'], phase=case['phase'], seed=case['seed'],
            beta=case['beta'], init_min_cosine=min(float(r['cosine']) for r in initial),
            trained_min_cosine=min(float(r['cosine']) for r in trained),
            init_max_norm_delta=max(float(r['symmetric_norm_delta']) for r in initial),
            trained_max_norm_delta=max(float(r['symmetric_norm_delta']) for r in trained),
            failed_layer_batches=sum(not flag(r['gradient_fidelity_passed']) for r in raw),
            layer_batches=len(raw), gradient_passed=case['gradient_fidelity_all_passed'],
            equilibrium_passed=case['equilibrium_residual_all_passed'],
            result=case['result'], result_sha256=case['result_sha256']))
    grouped = defaultdict(list)
    keys = ['phase', 'architecture', 'model_seed', 'injected_beta', 'checkpoint_role', 'cohort', 'parameter_name']
    for row in all_rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    distribution = []
    for key, rows in grouped.items():
        cosines = np.array([float(r['cosine']) for r in rows])
        distribution.append(dict(zip(keys, key), count=len(rows), minimum_cosine=float(cosines.min()),
            fifth_percentile_cosine=float(np.quantile(cosines, .05)), median_cosine=float(np.median(cosines)),
            maximum_norm_delta=max(float(r['symmetric_norm_delta']) for r in rows),
            cosine_failures=sum(float(r['cosine']) < .99 for r in rows),
            norm_failures=sum(float(r['symmetric_norm_delta']) > .1 for r in rows),
            combined_failures=sum(not flag(r['gradient_fidelity_passed']) for r in rows)))
    write_csv(analysis / 'case_summary.csv', case_rows)
    write_csv(analysis / 'layer_batch_metrics.csv', all_rows)
    write_csv(analysis / 'layer_cohort_summary.csv', distribution)
    failures = [r for r in all_rows if not flag(r['gradient_fidelity_passed'])]
    if failures:
        write_csv(analysis / 'failed_comparisons.csv', failures)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    colors = {'reconstructed_initialization':'#2676b9', 'best_validation':'#d77c18'}
    labels = {'reconstructed_initialization':'Initialization', 'best_validation':'Trained checkpoint'}
    for col, architecture in enumerate(('conv1', 'conv2', 'conv3')):
        for role, color in colors.items():
            minima = []; maxima = []
            for beta in (100, 200, 300):
                rr = [r for r in all_rows if r['phase']=='selection' and r['architecture']==architecture
                      and r['checkpoint_role']==role and float(r['injected_beta'])==beta]
                by_batch = defaultdict(list)
                for r in rr: by_batch[int(r['batch_index'])].append(r)
                cosine = [min(float(r['cosine']) for r in batch) for batch in by_batch.values()]
                norm = [max(float(r['symmetric_norm_delta']) for r in batch) for batch in by_batch.values()]
                jitter = np.linspace(-7, 7, len(cosine))
                axes[0,col].scatter(beta+jitter, cosine, color=color, s=8, alpha=.35)
                axes[1,col].scatter(beta+jitter, norm, color=color, s=8, alpha=.35)
                minima.append(min(cosine)); maxima.append(max(norm))
            axes[0,col].plot([100,200,300], minima, 'o-', color=color, label=labels[role])
            axes[1,col].plot([100,200,300], maxima, 'o-', color=color)
        axes[0,col].axhline(.99, color='black', linestyle='--', linewidth=1)
        axes[1,col].axhline(.1, color='black', linestyle='--', linewidth=1)
        axes[0,col].set_title(architecture.capitalize())
        axes[1,col].set_xlabel('Injected beta')
        axes[1,col].set_xticks([100,200,300])
        for ax in axes[:,col]:
            ax.grid(alpha=.2); ax.spines[['top','right']].set_visible(False)
    axes[0,0].set_ylabel('Worst-layer cosine')
    axes[1,0].set_ylabel('Largest norm mismatch')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Baseline beta selection, seed 0: 36 batches at initialization and the trained checkpoint\nDots: batches; solid lines: worst observed batch; dashed lines: qualification thresholds', fontsize=11)
    fig.tight_layout()
    fig.savefig(analysis / 'baseline_beta_audit.png', dpi=180)
    fig.savefig(analysis / 'baseline_beta_audit.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for arch in ('conv1', 'conv2'):
        rr = sorted([r for r in case_rows if r['phase']=='confirmation' and r['architecture']==arch], key=lambda r:r['seed'])
        if not rr: continue
        label = f"{arch.capitalize()}, beta {rr[0]['beta']}"
        axes[0].plot([r['seed'] for r in rr], [min(r['init_min_cosine'],r['trained_min_cosine']) for r in rr], 'o-', label=label)
        axes[1].plot([r['seed'] for r in rr], [max(r['init_max_norm_delta'],r['trained_max_norm_delta']) for r in rr], 'o-', label=label)
    axes[0].axhline(.99, color='black', linestyle='--', linewidth=1)
    axes[1].axhline(.1, color='black', linestyle='--', linewidth=1)
    axes[0].set_ylabel('Worst cosine')
    axes[1].set_ylabel('Largest norm mismatch')
    for ax in axes:
        ax.set_xticks([0,1,2]); ax.set_xlabel('Model seed'); ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle('Frozen-candidate confirmation: 32 reserved batches plus 4 historical batches')
    fig.tight_layout()
    fig.savefig(analysis / 'baseline_beta_confirmation.png', dpi=180)
    fig.savefig(analysis / 'baseline_beta_confirmation.pdf')
    plt.close(fig)
    summary = dict(study_id=OUT.name, selection_cases=9, confirmation_cases=expected_confirmation,
                   replay_count=sum(c['replay_count'] for c in cases),
                   layer_comparison_count=len(all_rows), failed_comparisons=len(failures),
                   decisions=execution['decisions'], source_result_sha256=source_hashes,
                   all_bundles_validated=True, source_bytes_unchanged=True,
                   official_test_read=False, optimizer_steps_applied=False,
                   historical_examples=64, new_selection_examples=512, new_confirmation_examples=512)
    regression = json.loads((OUT / 'historical_regression_comparison.json').read_text())
    summary['historical_regression'] = {k:v for k,v in regression.items() if k != 'rows'}
    (analysis / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n')
    lines = ['# Baseline beta audit: 100, 200, 300', '',
        'Completed read-only clean wide-range Adam diagnostic, 2026-09-14. '
        'The other schemes, learning rates and training configs remain unchanged.', '',
        'Each selection case uses model seed 0, initialization and the BPTT best checkpoint, '
        'with 32 new batches of 16 plus the original four batches. Thresholds are every-layer/batch '
        'cosine >= .99 and symmetric norm difference <= .10. Equilibrium residual qualification is separate.', '',
        '| Architecture | Beta | Worst initial cosine | Worst trained cosine | Failed layer/batches | Gradient gate | Equilibrium gate |',
        '|---|---:|---:|---:|---:|---|---|']
    for row in sorted(case_rows, key=lambda r:(r['architecture'],r['beta'],r['seed'])):
        if row['phase']!='selection': continue
        lines.append(f"| {row['architecture']} | {row['beta']} | {row['init_min_cosine']:.6f} | "
                     f"{row['trained_min_cosine']:.6f} | {row['failed_layer_batches']}/{row['layer_batches']} | "
                     f"{'pass' if row['gradient_passed'] else 'fail'} | {'pass' if row['equilibrium_passed'] else 'fail'} |")
    lines += ['', '## Selection and confirmation', '']
    for arch, decision in sorted(execution['decisions'].items()):
        lines.append(f"- {arch}: candidate entering confirmation **{decision['selected_beta'] if decision['selected_beta'] is not None else 'none'}**; "
                     f"state `{decision['state']}`.")
    lines += ['', '| Architecture | Candidate beta | Seed | Worst cosine | Largest norm mismatch | Gradient gate |',
              '|---|---:|---:|---:|---:|---|']
    for row in sorted(case_rows, key=lambda r:(r['architecture'],r['seed'])):
        if row['phase'] != 'confirmation': continue
        lines.append(f"| {row['architecture']} | {row['beta']} | {row['seed']} | "
                     f"{min(row['init_min_cosine'],row['trained_min_cosine']):.6f} | "
                     f"{max(row['init_max_norm_delta'],row['trained_max_norm_delta']):.6f} | "
                     f"{'pass' if row['gradient_passed'] else 'fail'} |")
    lines += ['', 'Conv2 beta 100 passes the numerical selection and all three confirmation seeds. '
              'Conv1 beta 300 fails three trained-checkpoint layer/batch comparisons on seed 2; '
              'beta 100 and 200 pass seed-0 selection but were not subsequently confirmed across seeds in this round. '
              'Conv3 fails all three candidates at initialization and retains a separate trained free-state residual failure. '
              'No training beta has been changed.', '',
              f"Historical check: all {regression['matched_rows']} original baseline beta-100 layer comparisons "
              f"retain identical input payloads; maximum cosine change {regression['max_cosine_abs_delta']:.3g} "
              f"and norm-mismatch change {regression['max_norm_delta_abs_delta']:.3g}."]
    lines += ['', f"Coverage: **9/9 selection cases and {expected_confirmation}/{expected_confirmation} conditionally required confirmation cases**; "
              f"{summary['replay_count']} checkpoint/batch replays and {len(all_rows)} layer comparisons. All included canonical bundles validate.", '',
              'Only a passing candidate is tested on the reserved confirmation cohort and seeds 0/1/2. '
              'An architecture with no passing candidate has no confirmed replacement beta. '
              'Confirmation failures are retained without trying another beta on the same cohort. '
              'These are sampled gradient checks, not proof of accuracy, full-training stability or equivalence. '
              'The official test split was not read and no training was launched.', '',
              'The first preparation smoke stopped before computation because its inherited runtime path '
              'pointed at the working checkout. Its log/configs are retained; the corrected smoke and '
              'production use the hash-verified frozen runtime and version-2 configs.', '',
              '![Baseline beta audit](baseline_beta_audit_20260914.png)', '',
              '![Frozen-candidate confirmation](baseline_beta_confirmation_20260914.png)', '',
              '[Per-case measurements](baseline_beta_audit_20260914.csv) · '
              '[Per-layer/cohort distributions](baseline_beta_layer_cohort_summary_20260914.csv) · '
              '[Full local evidence](../results/eqprop-beta-selection-audit-20260914-v1/analysis/) · '
              '[Execution plan](../docs/eqprop_baseline_beta_audit_20260914.md)', '']
    report = '\n'.join(lines)
    paper = ROOT / 'paper_ready_results'
    (paper / 'baseline_beta_audit_20260914.md').write_text(report)
    # Keep links correct in the standalone study report as well.
    (analysis / 'report.md').write_text(report.replace('baseline_beta_audit_20260914.png','baseline_beta_audit.png')
        .replace('baseline_beta_confirmation_20260914.png','baseline_beta_confirmation.png')
        .replace('baseline_beta_audit_20260914.csv','case_summary.csv')
        .replace('baseline_beta_layer_cohort_summary_20260914.csv','layer_cohort_summary.csv')
        .replace('../results/eqprop-beta-selection-audit-20260914-v1/analysis/','./')
        .replace('../docs/eqprop_baseline_beta_audit_20260914.md','../../../docs/eqprop_baseline_beta_audit_20260914.md'))
    for src,dst in [('case_summary.csv','baseline_beta_audit_20260914.csv'),
                    ('layer_cohort_summary.csv','baseline_beta_layer_cohort_summary_20260914.csv'),
                    ('summary.json','baseline_beta_audit_20260914.json'),
                    ('baseline_beta_audit.png','baseline_beta_audit_20260914.png'),
                    ('baseline_beta_audit.pdf','baseline_beta_audit_20260914.pdf'),
                    ('baseline_beta_confirmation.png','baseline_beta_confirmation_20260914.png'),
                    ('baseline_beta_confirmation.pdf','baseline_beta_confirmation_20260914.pdf')]:
        shutil.copy2(analysis/src, paper/dst)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
