"""Join existing clean training outcomes to their static beta diagnostics."""
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper_ready_results'


def key(arch, scheme, beta):
    return arch, scheme, float(f'{float(beta):.12g}')


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    measurements = {}
    for grouping, filename in (
        ('matrix', 'beta_rule_comparison_20260918_layer_metrics.csv'),
        ('whole', 'beta_rule_comparison_20260918_whole_gradient_metrics.csv'),
    ):
        grouped = defaultdict(list)
        with (OUT / filename).open() as stream:
            for r in csv.DictReader(stream):
                grouped[key(r['architecture'], r['scheme'], r['injected_beta'])].append(r)
        measurements[grouping] = grouped
    candidates = {}
    for study in ('eqprop-beta-training-stability-20260918-v1', 'eqprop-layerwise-beta-training-20260919-v1'):
        data = json.loads((ROOT / 'results' / study / 'analysis.json').read_text())
        for r in data['rows']:
            k = key(r['architecture'], r['scheme'], r['injected_beta'])
            candidates.setdefault(k, dict(architecture=k[0], scheme=k[1], beta=k[2],
                bundle=r['bundle'], origin=study))
    for arch in ('conv1', 'conv2', 'conv3'):
        for scheme in ('baseline', 'ours', 'legacy'):
            run = OUT / 'bundles/table2_wide_ep' / arch / scheme / 'seed0'
            cfg = json.loads((run / 'config.used.json').read_text())
            k = key(arch, scheme, cfg['eqprop']['injected_beta_B'])
            candidates.setdefault(k, dict(architecture=arch, scheme=scheme, beta=k[2],
                bundle=str(run.relative_to(ROOT)), origin='historical_control_first10'))
    rows = []
    distributions = []
    for k, row in sorted(candidates.items()):
        run = ROOT / row['bundle']
        cfg = json.loads((run / 'config.used.json').read_text())
        assert cfg['seed'] == 0 and cfg['eqprop']['endpoint_read_noise_std'] == 0
        assert cfg['evaluation']['official_test']['policy'] == 'disabled'
        status = json.loads((run / 'status.json').read_text())
        records = [json.loads(x) for x in (run / 'metrics.jsonl').read_text().splitlines()]
        epochs = [r for r in records if r.get('kind') == 'epoch' and r['epoch'] <= 10]
        values = [100 * r['metrics']['validation_accuracy'] for r in epochs]
        successful = status['state'] == 'complete' and len(epochs) == 10
        row.update(ten_epoch_complete=successful, epochs_completed=len(epochs),
                   final_validation_pct=values[-1] if successful else '',
                   best_observed_validation_pct=max(values) if values else '',
                   early_stable=successful and max(values)-values[-1] < 5-1e-10,
                   training_state=status['state'])
        for grouping, groups in measurements.items():
            rs = groups[k]
            assert len(rs) == 72 * (int(k[0][-1])+1 if grouping == 'matrix' else 1)
            assert len({(r['checkpoint_role'],r['batch_index']) for r in rs}) == 72
            assert all(int(r['T']) == int(k[0][-1])*2+2 and r['T'] == r['K'] for r in rs)
            for label, field, reduce in (
                ('minimum_cosine', 'cosine', min),
                ('maximum_norm_mismatch', 'symmetric_norm_delta', max),
                ('maximum_relative_l2_error', 'relative_l2_difference_over_bptt', max),
            ):
                xs = [float(r[field]) for r in rs]
                assert all(map(math.isfinite, xs))
                row[f'{grouping}_{label}'] = reduce(xs)
            if grouping == 'matrix':
                replays = defaultdict(list)
                for r in rs:
                    replays[r['checkpoint_role'], int(r['batch_index'])].append(float(r['cosine']))
                roles = sorted({role for role, batch in replays})
                assert len(roles) == 2
                for role in roles:
                    minima = [min(xs) for (checkpoint, batch), xs in replays.items() if checkpoint == role]
                    assert len(minima) == 36
                    distributions.append(dict(architecture=k[0], scheme=k[1], beta=k[2], checkpoint_role=role,
                        replay_count=len(minima), minimum_matrix_cosine=min(minima),
                        median_batch_minimum_cosine=statistics.median(minima), maximum_batch_minimum_cosine=max(minima),
                        batches_all_matrices_above_090=sum(x > .90 for x in minima),
                        batches_all_matrices_above_095=sum(x > .95 for x in minima),
                        early_stable=row['early_stable']))
        rows.append(row)
    assert len(rows) == 23
    rules = []
    for grouping in ('matrix', 'whole'):
        for threshold in (.90, .95, .99):
            for norm_gate in (False, True):
                accepted = [r for r in rows if r[f'{grouping}_minimum_cosine'] > threshold and
                            (not norm_gate or r[f'{grouping}_maximum_norm_mismatch'] <= .10)]
                rules.append(dict(grouping=grouping, threshold=threshold, norm_gate=norm_gate,
                    accepted=len(accepted), accepted_stable=sum(r['early_stable'] for r in accepted),
                    accepted_failed=sum(r['training_state']=='failed' for r in accepted),
                    rejected_stable=sum(r['early_stable'] for r in rows if r not in accepted)))
    write_csv(OUT / 'beta_protocol_training_evidence_20260919.csv', rows)
    write_csv(OUT / 'beta_protocol_rule_outcomes_20260919.csv', rules)
    write_csv(OUT / 'beta_protocol_batch_distributions_20260919.csv', distributions)
    payload = dict(distinct_beta_settings=len(rows), stable=sum(r['early_stable'] for r in rows),
                   failed=sum(r['training_state']=='failed' for r in rows), rows=rows, rules=rules,
                   interpretation='Retrospective selected one-seed evidence; rule outcomes are not unbiased predictive-accuracy estimates.')
    (ROOT / 'results/eqprop-beta-refinement-20260919-v1/training_evidence.json').write_text(json.dumps(payload, indent=2)+'\n')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('rows','rules')}))


if __name__ == '__main__':
    main()
