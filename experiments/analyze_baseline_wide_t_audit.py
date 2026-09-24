"""Validate and tabulate the fixed-beta/K Conv3 free-phase T experiment."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.reporting import validate_run
from experiments.run_baseline_wide_t_audit import ROOT, OUT, TS, checked, read, sha, write


def flag(value):
    return str(value).lower() == 'true'


def csv_write(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    execution = read(OUT / 'execution.json')
    if execution['state'] != 'complete':
        raise ValueError('The declared study must reach terminal coverage first.')
    cases = execution['runs']
    assert sorted(r['T'] for r in cases if r['phase'] == 'selection') == list(TS)
    confirmations = [r for r in cases if r['phase'] == 'confirmation']
    assert len(confirmations) == (0 if execution['selected_T'] is None else 3)
    if confirmations:
        assert sorted(r['seed'] for r in confirmations) == [0, 1, 2]
    checked(execution['smoke'], smoke=True)
    for case in cases:
        result = checked(case)
        assert result['result_sha256'] == case['result_sha256']
    reference = read(OUT / 'reference.json')
    directory = (ROOT / reference['result']).parent
    if errors := validate_run(directory):
        raise ValueError(errors)
    assert sha(directory / 'result.json') == reference['result_sha256']
    reference.update(phase='reference', seed=0,
                     **read(directory / 'result.json')['terminal_metrics'])
    all_cases = [reference, *cases]
    analysis = OUT / 'analysis'
    analysis.mkdir(exist_ok=True)
    case_rows, layer_rows, residual_rows = [], [], []
    reference_payloads = None
    for case in all_cases:
        directory = (ROOT / case['result']).parent
        raw = list(csv.DictReader((directory / 'layer_metrics.csv').open()))
        residual = list(csv.DictReader((directory / 'equilibrium_residuals.csv').open()))
        assert len(raw) == 288 and len(residual) == 1152
        payloads = {(r['checkpoint_role'], int(r['batch_index']), r['parameter_name']):
                    (r['batch_payload_sha256'], r['batch_source_indices_sha256']) for r in raw}
        if case['phase'] == 'reference':
            reference_payloads = payloads
        elif case['phase'] == 'selection':
            assert payloads == reference_payloads, 'Unmatched T comparison inputs.'
        row = dict(phase=case['phase'], seed=case['seed'], T=case['T'], K=8, beta=100,
                   gradient_pass=case['gradient_fidelity_all_passed'],
                   equilibrium_pass=case['equilibrium_residual_all_passed'],
                   combined_pass=case['unqualified_launch_gate_all_passed'],
                   failed_layer_batches=sum(not flag(r['gradient_fidelity_passed']) for r in raw))
        for role, prefix in [('reconstructed_initialization', 'init'), ('best_validation', 'trained')]:
            layers = [r for r in raw if r['checkpoint_role'] == role]
            states = [r for r in residual if r['checkpoint_role'] == role]
            row.update({f'{prefix}_min_cosine': min(float(r['cosine']) for r in layers),
                f'{prefix}_max_norm_delta': max(float(r['symmetric_norm_delta']) for r in layers),
                f'{prefix}_failed_layer_batches': sum(not flag(r['gradient_fidelity_passed']) for r in layers),
                f'{prefix}_max_free_residual_p90': max(float(r['selected_max_p90']) for r in states if r['phase'] == 'post_T_free'),
                f'{prefix}_max_endpoint_residual_p90': max(float(r['selected_max_p90']) for r in states),
                f'{prefix}_failed_residual_checks': sum(not flag(r['gate_passed']) for r in states)})
        row.update(result=case['result'], result_sha256=case['result_sha256'])
        case_rows.append(row)
        for r in raw:
            r.update(audit_phase=case['phase'], model_seed=case['seed'],
                     cohort='historical' if int(r['batch_index']) < 4 else case['phase'])
            layer_rows.append(r)
        for r in residual:
            r.update(audit_phase=case['phase'], model_seed=case['seed'])
            residual_rows.append(r)
    csv_write(analysis / 'case_summary.csv', case_rows)
    csv_write(analysis / 'layer_batch_metrics.csv', layer_rows)
    csv_write(analysis / 'equilibrium_residuals.csv', residual_rows)
    groups = defaultdict(list)
    keys = ('audit_phase', 'model_seed', 'T', 'checkpoint_role', 'cohort', 'parameter_name')
    for row in layer_rows:
        groups[tuple(row[k] for k in keys)].append(row)
    distributions = []
    for group, rows in groups.items():
        cosines = np.array([float(r['cosine']) for r in rows])
        distributions.append(dict(zip(keys, group), count=len(rows),
            min_cosine=float(cosines.min()), p05_cosine=float(np.quantile(cosines, .05)),
            median_cosine=float(np.median(cosines)),
            max_norm_delta=max(float(r['symmetric_norm_delta']) for r in rows),
            failed=sum(not flag(r['gradient_fidelity_passed']) for r in rows)))
    csv_write(analysis / 'layer_cohort_summary.csv', distributions)
    csv_write(analysis / 'failed_comparisons.csv',
              [r for r in layer_rows if not flag(r['gradient_fidelity_passed'])])
    selection = [r for r in case_rows if r['phase'] != 'confirmation']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for prefix, label, color in [('init', 'Initialization', '#2676b9'),
                                  ('trained', 'Trained checkpoint', '#d77c18')]:
        xs = [r['T'] for r in selection]
        for ax, suffix in zip(axes, ('min_cosine', 'max_norm_delta', 'max_free_residual_p90')):
            ax.plot(xs, [r[f'{prefix}_{suffix}'] for r in selection], 'o-', color=color, label=label)
    for ax, threshold, label in zip(axes, (.99, .1, .01),
        ('Worst gradient cosine', 'Largest gradient norm mismatch', 'Worst free-state residual p90')):
        ax.axhline(threshold, color='black', linestyle='--', linewidth=1)
        ax.set(xlabel='Free-phase T (K=8 fixed)', ylabel=label, xticks=[8, *TS])
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[2].set_yscale('log')
    axes[0].legend(fontsize=8)
    fig.suptitle('Wide Conv3 baseline, beta=100: only free-phase T changes\nSeed 0; same 36 batches at initialization and trained checkpoint; T=8 reuses the prior audit', fontsize=11)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(analysis / f'baseline_wide_t_audit.{suffix}', dpi=180)
    plt.close(fig)
    summary = dict(study_id=OUT.name, execution=execution, cases=case_rows,
        reference_included=True, new_formal_cases=len(cases),
        new_checkpoint_batch_replays=72*len(cases), new_layer_comparisons=288*len(cases),
        new_failed_layer_comparisons=sum(r['failed_layer_batches'] for r in case_rows if r['phase'] != 'reference'),
        source_sha256=sha(__file__), input_payloads_identical_across_T=True,
        official_test_read=False, training_started=False)
    write(analysis / 'summary.json', summary)
    for source, dest in [('case_summary.csv', 'baseline_wide_t_audit_20260916.csv'),
                         ('summary.json', 'baseline_wide_t_audit_20260916.json'),
                         ('layer_cohort_summary.csv', 'baseline_wide_t_layer_summary_20260916.csv'),
                         ('baseline_wide_t_audit.png', 'baseline_wide_t_audit_20260916.png'),
                         ('baseline_wide_t_audit.pdf', 'baseline_wide_t_audit_20260916.pdf')]:
        shutil.copy2(analysis / source, ROOT / 'paper_ready_results' / dest)
    print(json.dumps(case_rows, indent=2))


if __name__ == '__main__':
    main()
