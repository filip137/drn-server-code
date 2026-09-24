"""Audit retained full-budget runs against two provisional Conv1 beta interpretations.

Reads local metadata only. This does not select beta, submit jobs, or promote
validation results to paper evidence.
"""
import csv
import json
import math
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STEM = 'beta_stability_protocol_run_audit_20260921'
SCHEMES = ('baseline', 'ours', 'legacy')
SIGMAS = (1e-5, 3e-5, 1e-4, 3e-4, 5e-4)


def read_csv(name):
    with (HERE / name).open() as stream:
        return list(csv.DictReader(stream))


def main():
    p99 = {(r['architecture'], r['scheme']): float(r['selected_beta'])
           for r in read_csv('beta_rule_comparison_20260918_selections.csv')
           if r['checkpoint_role'] == 'both' and r['grouping'] == 'layerwise'
           and float(r['cosine_threshold']) == .99 and r['norm_gate'] == 'False'}
    p95 = {(r['architecture'], r['scheme']): float(r['beta'])
           for r in read_csv('beta_refinement_20260919_selections.csv')
           if r['architecture'] == 'conv3' and float(r['threshold']) == .95}
    existing = []

    def add(row, scheme, seed, sigma, status, tracker):
        if status != 'complete_collected_validated':
            return
        bundle = HERE / row['collected_result']
        cfg = json.loads((bundle.parent / 'config.used.json').read_text())
        assert bundle.is_file()
        beta = float(cfg['eqprop']['injected_beta_B'])
        expected_epochs = 10 if row['architecture'] == 'conv1' else 30
        assert int(row['epochs']) == expected_epochs
        if row.get('completed_epochs'):
            assert int(row['completed_epochs']) == expected_epochs
        model = cfg['model_base']
        iterations = {'conv1': 4, 'conv2': 6, 'conv3': 8}[row['architecture']]
        assert model['num_iterations_inference'] == model['num_iterations_training'] == iterations
        assert model['weight_min'] == 0 and model['weight_max'] == 100
        assert all(value == 0 for name, value in cfg['learning_rates_by_parameter'].items()
                   if name.startswith('Bias_'))
        existing.append(dict(architecture=row['architecture'], scheme=scheme,
                             seed=int(seed), sigma=float(sigma), beta=beta,
                             bundle=str(bundle.parent.relative_to(ROOT)), tracker=tracker))

    for row in read_csv('current_contract_run_status.csv'):
        if row['table'] == 'table2' and row['algorithm'] == 'EP' and float(row['G_max']) == 100:
            add(row, row['scheme'], row['model_seed'], 0, row['training_status'],
                'current_contract_run_status.csv')
    for name in ('read_noise_run_status_20260914.csv', 'baseline_read_noise_run_status_20260916.csv'):
        for row in read_csv(name):
            add(row, row.get('scheme', 'baseline'), row.get('seed', 0), row['sigma'], row['status'], name)

    coverage = []
    summaries = {}
    for interpretation in ('latest_p99_div10', 'historical_conv1'):
        targets = {(arch, scheme): (p99[arch, scheme] / 10 if arch == 'conv1'
                                   else p99[arch, scheme] if arch == 'conv2'
                                   else p95[arch, scheme])
                   for arch in ('conv1', 'conv2', 'conv3') for scheme in SCHEMES}
        if interpretation == 'historical_conv1':
            targets.update({('conv1', scheme): beta
                            for scheme, beta in zip(SCHEMES, (100, 30, 3))})
        for (arch, scheme), beta in targets.items():
            for seed, sigma in [(s, 0) for s in (0, 1, 2)] + [(0, s) for s in SIGMAS]:
                matches = [r for r in existing if r['architecture'] == arch and r['scheme'] == scheme
                           and r['seed'] == seed and r['sigma'] == sigma
                           and math.isclose(r['beta'], beta, rel_tol=1e-12)]
                assert len(matches) <= 1
                already_declared = arch == 'conv3' and seed == 0 and sigma == 0
                state = 'complete_beta_match' if matches else ('already_declared_unverified' if already_declared else 'missing')
                coverage.append(dict(interpretation=interpretation, architecture=arch, scheme=scheme,
                                     injected_beta=beta, seed=seed, training_sigma=sigma,
                                     epochs=10 if arch == 'conv1' else 30, state=state,
                                     existing_bundle=matches[0]['bundle'] if matches else '',
                                     existing_tracker=matches[0]['tracker'] if matches else '',
                                     existing_plan='docs/eqprop_conv3_p95_read_noise_1em3_plan_20260921.md'
                                     if already_declared and not matches else ''))
        selected = [r for r in coverage if r['interpretation'] == interpretation]
        assert len(selected) == 72
        summaries[interpretation] = dict(Counter(r['state'] for r in selected))

    with (HERE / f'{STEM}_coverage.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    print(json.dumps(summaries, indent=2))
    print(HERE / f'{STEM}_coverage.csv')


if __name__ == '__main__':
    main()
