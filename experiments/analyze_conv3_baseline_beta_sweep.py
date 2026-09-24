"""Collect the fixed-T8/K8 Conv3 beta sweep and its reserved confirmation."""
from __future__ import annotations

from collections import defaultdict
import csv
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.run_conv3_baseline_beta_sweep import ROOT, OUT, BETAS, checked, reference, read, sha, write


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
        raise ValueError('Wait for terminal declared coverage.')
    cases = execution['runs']
    assert sorted(r['beta'] for r in cases if r['phase'] == 'selection') == list(BETAS)
    confirmations = [r for r in cases if r['phase'] == 'confirmation']
    assert len(confirmations) == (0 if execution['selected_beta'] is None else 3)
    if confirmations:
        assert sorted(r['seed'] for r in confirmations) == [0, 1, 2]
        assert {r['beta'] for r in confirmations} == {execution['selected_beta']}
    checked(execution['smoke'], smoke=True)
    for case in cases:
        assert checked(case)['result_sha256'] == case['result_sha256']
    all_cases = [reference(), *cases]
    analysis = OUT / 'analysis'
    analysis.mkdir(exist_ok=True)
    summaries, all_layers, all_residuals = [], [], []
    reference_payloads, reference_free = None, None
    for case in all_cases:
        directory = (ROOT / case['result']).parent
        layers = list(csv.DictReader((directory / 'layer_metrics.csv').open()))
        states = list(csv.DictReader((directory / 'equilibrium_residuals.csv').open()))
        assert len(layers) == 288 and len(states) == 1152
        assert all(int(r['T']) == int(r['K']) == 8 for r in layers + states)
        assert all(float(r['injected_beta']) == case['beta'] for r in layers)
        payloads = {(r['checkpoint_role'], int(r['batch_index']), r['parameter_name']):
                    (r['batch_payload_sha256'], r['batch_source_indices_sha256']) for r in layers}
        free = {(r['checkpoint_role'], int(r['batch_index']), r['state_layer_name']):
                float(r['selected_max_p90']) for r in states if r['phase'] == 'post_T_free'}
        if case['phase'] == 'reference':
            reference_payloads, reference_free = payloads, free
        elif case['phase'] == 'selection':
            assert payloads == reference_payloads, 'Mismatched selection inputs.'
            assert free.keys() == reference_free.keys()
            assert all(np.isclose(free[k], reference_free[k], rtol=1e-12, atol=1e-14) for k in free), 'Free states differ between betas.'
        row = dict(phase=case['phase'], seed=case['seed'], T=8, K=8, beta=case['beta'],
            gradient_pass=case['gradient_fidelity_all_passed'],
            equilibrium_pass=case['equilibrium_residual_all_passed'],
            combined_pass=case['unqualified_launch_gate_all_passed'],
            failed_layer_batches=sum(not flag(r['gradient_fidelity_passed']) for r in layers))
        for role, prefix in [('reconstructed_initialization', 'init'), ('best_validation', 'trained')]:
            ll = [r for r in layers if r['checkpoint_role'] == role]
            ss = [r for r in states if r['checkpoint_role'] == role]
            row.update({f'{prefix}_min_cosine': min(float(r['cosine']) for r in ll),
                f'{prefix}_max_norm_delta': max(float(r['symmetric_norm_delta']) for r in ll),
                f'{prefix}_failed_layer_batches': sum(not flag(r['gradient_fidelity_passed']) for r in ll),
                f'{prefix}_max_free_residual_p90': max(float(r['selected_max_p90']) for r in ss if r['phase'] == 'post_T_free'),
                f'{prefix}_max_endpoint_residual_p90': max(float(r['selected_max_p90']) for r in ss),
                f'{prefix}_failed_residual_checks': sum(not flag(r['gate_passed']) for r in ss)})
        row.update(result=case['result'], result_sha256=case['result_sha256'])
        summaries.append(row)
        for r in layers:
            r.update(audit_phase=case['phase'], model_seed=case['seed'],
                     cohort='historical' if int(r['batch_index']) < 4 else case['phase'])
            all_layers.append(r)
        for r in states:
            r.update(audit_phase=case['phase'], model_seed=case['seed'], sweep_beta=case['beta'])
            all_residuals.append(r)
    csv_write(analysis / 'case_summary.csv', summaries)
    csv_write(analysis / 'layer_batch_metrics.csv', all_layers)
    csv_write(analysis / 'equilibrium_residuals.csv', all_residuals)
    csv_write(analysis / 'failed_comparisons.csv', [r for r in all_layers if not flag(r['gradient_fidelity_passed'])])
    keys = ('audit_phase', 'model_seed', 'injected_beta', 'checkpoint_role', 'cohort', 'parameter_name')
    groups = defaultdict(list)
    for row in all_layers:
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
    selection = sorted([r for r in summaries if r['phase'] != 'confirmation'], key=lambda r:r['beta'])
    confirmation = sorted([r for r in summaries if r['phase'] == 'confirmation'], key=lambda r:r['seed'])
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for index, (rows, field) in enumerate(((selection, 'beta'), (confirmation, 'seed'))):
        for prefix, label, color in [('init', 'Initialization', '#2676b9'), ('trained', 'Trained checkpoint', '#d77c18')]:
            for ax, suffix in zip(axes[index], ('min_cosine', 'max_norm_delta')):
                ax.plot([r[field] for r in rows], [r[f'{prefix}_{suffix}'] for r in rows], 'o-', color=color, label=label)
        for ax, threshold, label in zip(axes[index], (.99, .1), ('Worst gradient cosine', 'Largest gradient norm mismatch')):
            ax.axhline(threshold, color='black', linestyle='--', linewidth=1)
            ax.set(xlabel='Injected beta' if index == 0 else 'Model seed', ylabel=label,
                xticks=[10, 30, 50, 75, 100] if index == 0 else [0, 1, 2])
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
        axes[index, 0].set_title('Selection: seed 0, same 36 batches' if index == 0 else
            f'Confirmation: beta={execution["selected_beta"]}, reserved 36 batches')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Wide Conv3 baseline beta sweep: T=8, K=8 fixed\nGradient qualification; the separate T=8 equilibrium caveat remains', fontsize=12)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(analysis / f'conv3_baseline_beta_tk8.{suffix}', dpi=180)
    plt.close(fig)
    summary = dict(study_id=OUT.name, execution=execution, cases=summaries,
        new_formal_cases=len(cases), new_checkpoint_batch_replays=72*len(cases),
        new_layer_comparisons=288*len(cases),
        new_failed_layer_comparisons=sum(r['failed_layer_batches'] for r in summaries if r['phase'] != 'reference'),
        reference_beta100_reused=True, T=8, K=8,
        selection_payloads_identical=True, free_residuals_unchanged_by_beta=True,
        analysis_source_sha256=sha(__file__), official_test_read=False, training_started=False)
    write(analysis / 'summary.json', summary)
    for source, dest in [('case_summary.csv', 'conv3_baseline_beta_tk8_20260916.csv'),
                         ('summary.json', 'conv3_baseline_beta_tk8_20260916.json'),
                         ('layer_cohort_summary.csv', 'conv3_baseline_beta_tk8_layers_20260916.csv'),
                         ('conv3_baseline_beta_tk8.png', 'conv3_baseline_beta_tk8_20260916.png'),
                         ('conv3_baseline_beta_tk8.pdf', 'conv3_baseline_beta_tk8_20260916.pdf')]:
        shutil.copy2(analysis / source, ROOT / 'paper_ready_results' / dest)
    print('Validated and promoted',len(cases),'new formal cases; decision:',execution['decision'])


if __name__ == '__main__':
    main()
