"""Tabulate existing Conv3 calibration/accuracy evidence; run no simulations."""
import csv
import json
import math
import statistics
from pathlib import Path

from experiments.reporting import sha256_file, validate_run

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper_ready_results'
STEM = 'conv3_p90_p99_existing_20260921'
SCHEMES = ('baseline', 'legacy', 'ours')
PARAMETERS = ('ConvWeight_0', 'ConvWeight_1', 'ConvWeight_2', 'DenseWeight_0')
LAYERS = ('Layer_1', 'Layer_2', 'Layer_3', 'Layer_4')


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    grid = read_csv(OUT / 'beta_refinement_20260919_metrics.csv')
    progress = json.loads((ROOT / 'results/eqprop-beta-refinement-20260919-v1/local_conv3/progress.json').read_text())
    provenance = []
    cohort = None
    rows = []
    selections = []
    for threshold in (.90, .99):
        for scheme in SCHEMES:
            selected = max((r for r in grid if r['architecture'] == 'conv3'
                            and r['scheme'] == scheme and float(r['minimum_cosine']) > threshold),
                           key=lambda r: float(r['beta']))
            beta = float(selected['beta'])
            refined = [r for r in progress['runs'] if r['scheme'] == scheme and r['beta'] == beta]
            bundle = (Path(refined[0]['bundle']) if refined else
                      ROOT / 'results/eqprop-beta-rule-comparison-20260918-v1/production' /
                      f'conv3_{scheme}_beta{beta:g}'.replace('.', 'p'))
            errors = validate_run(bundle)
            assert not errors, (bundle, errors)
            terminal = json.loads((bundle / 'result.json').read_text())['terminal_metrics']
            assert terminal['replay_count'] == 72 and terminal['endpoint_read_noise_std'] == 0
            assert not terminal['optimizer_steps_applied'] and not terminal['official_test_read']
            assert terminal['source_bytes_unchanged'] and terminal['all_bias_tensors_exact_zero']
            layers = read_csv(bundle / 'layer_metrics.csv')
            assert min(float(r['cosine']) for r in layers) > threshold
            layers = [r for r in layers if r['checkpoint_role'] == 'reconstructed_initialization']
            displacements = [r for r in read_csv(bundle / 'state_displacement.csv')
                             if r['checkpoint_role'] == 'reconstructed_initialization']
            actual_cohort = sorted({(int(r['batch_index']), r['batch_payload_sha256'],
                                    r['batch_source_indices_sha256']) for r in layers})
            if cohort is None:
                cohort = actual_cohort
            assert cohort == actual_cohort and len(cohort) == 36
            selections.append(dict(threshold=threshold, scheme=scheme, beta=beta,
                                   origin=selected['origin'], minimum_cosine=float(selected['minimum_cosine'])))
            provenance.append(dict(bundle=str(bundle.relative_to(ROOT)), validated=True,
                                   hashes={n: sha256_file(bundle / n) for n in
                                           ('result.json', 'layer_metrics.csv', 'state_displacement.csv')}))
            for parameter, layer in zip(PARAMETERS, LAYERS):
                lr = [r for r in layers if r['parameter_name'] == parameter]
                assert len(lr) == 36 and all(float(r['endpoint_read_noise_std']) == 0 for r in lr)

                def pooled(phase, reference):
                    values = [r for r in displacements if r['state_layer_name'] == layer
                              and r['phase'] == phase and r['reference_kind'] == reference]
                    assert len(values) == 36 and all(r['outcome'] == 'ok' for r in values)
                    return math.sqrt(sum(float(r['delta_squared_sum']) for r in values) /
                                     sum(int(r['element_count']) for r in values))

                plus = pooled('positive', 'post_T_free')
                minus = pooled('negative', 'post_T_free')
                odd = pooled('positive_minus_negative', 'negative_phase') / 2
                rows.append(dict(threshold=threshold, scheme=scheme, beta=beta,
                                 parameter=parameter, state_layer=layer, batch_count=36,
                                 clean_cosine_min=min(float(r['cosine']) for r in lr),
                                 clean_cosine_median=statistics.median(float(r['cosine']) for r in lr),
                                 noisy_cosine_sigma1em3=None,
                                 physical_positive_free_rms=plus, physical_negative_free_rms=minus,
                                 physical_centered_half_difference_rms=odd,
                                 positive_readout_expected_rms_sigma1em3=math.sqrt(plus**2 + 1e-6),
                                 centered_readout_expected_rms_sigma1em3=math.sqrt(odd**2 + 1e-6 / 2),
                                 noisy_readout_values='analytic sqrt(expected MSE), not measured',
                                 source_bundle=str(bundle.relative_to(ROOT))))
    write_csv(OUT / f'{STEM}_layers.csv', rows)

    p90 = read_csv(OUT / 'conv3_p90_read_noise_1em3_20260920.csv')
    accuracies = []
    for r in p90:
        for path in (r['bundle'], r['clean_reference_bundle']):
            assert not validate_run(ROOT / path), path
        accuracies.append(dict(threshold=.9, scheme=r['scheme'], beta=float(r['beta']),
                               clean_epochs=30, clean_validation_pct=float(r['clean_validation_pct']),
                               noisy_epochs_completed=int(r['epochs_completed']),
                               noisy_final_validation_pct=float(r['final_validation_pct']) if r['final_validation_pct'] else None,
                               noisy_last_validation_pct=float(r['latest_validation_pct']) if r['latest_validation_pct'] else None,
                               noisy_state=r['state'], failure_detail=r['failure_detail'],
                               clean_bundle=r['clean_reference_bundle'], noisy_bundle=r['bundle']))
    baseline = next(r for r in read_csv(OUT / 'baseline_read_noise_run_status_20260916.csv')
                    if r['architecture'] == 'conv3' and float(r['beta']) == 10 and float(r['sigma']) == 0)
    baseline_path = OUT / baseline['collected_result']
    assert not validate_run(baseline_path.parent)
    baseline_metric = json.loads(baseline_path.read_text())['terminal_metrics']['validation']['final_accuracy'] * 100
    for r in selections:
        if r['threshold'] != .99:
            continue
        accuracies.append(dict(threshold=.99, scheme=r['scheme'], beta=r['beta'],
                               clean_epochs=30 if r['scheme'] == 'baseline' else None,
                               clean_validation_pct=baseline_metric if r['scheme'] == 'baseline' else None,
                               noisy_epochs_completed=None, noisy_final_validation_pct=None,
                               noisy_last_validation_pct=None, noisy_state='not measured', failure_detail='',
                               clean_bundle=str(baseline_path.parent.relative_to(ROOT)) if r['scheme'] == 'baseline' else '',
                               noisy_bundle=''))
    write_csv(OUT / f'{STEM}_accuracy.csv', accuracies)
    report = [
        '# Conv3: existing per-matrix cosine >0.90 versus >0.99 evidence', '',
        'Compiled 2026-09-21 without new replay or training. Seed 0, T=K=8, float64 centered frozen-current EqProp, '
        'weights [0,100], frozen zero biases. Noise comparison: sigma=0 versus 1e-3. Accuracies are ordinary-MNIST '
        '5,000-example validation measurements, not official-test results.', '',
        'Select the largest beta already measured in the combined original/refined grid that passes strictly for every '
        'weight matrix in every one of 36 batches at initialization and 36 at the saved BPTT checkpoint. '
        'No norm gate. The p90 choices are the trained refined choices. The p99 choices are a retrospective lookup '
        'in those existing measurements, not a new refined p99 search or training qualification. '
        'Ours beta0.987333678708 passes and exceeds the original-grid beta0.9; baseline10 and legacy0.1 remain '
        'coarse-grid choices. These are measured maxima, not proven global beta limits.', '',
        '## A. Initial per-layer gradient cosine', '',
        'Entries are **minimum / median** across the same 36 batches at initialization only. '
        'The four columns correspond to ConvWeight_0, ConvWeight_1, ConvWeight_2, DenseWeight_0.', '',
        '| Threshold | Scheme | Beta | Conv1 | Conv2 | Conv3 | Dense |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for s in selections:
        selected_rows = [r for r in rows if r['threshold'] == s['threshold'] and r['scheme'] == s['scheme']]
        report.append(f"| {s['threshold']:.2f} | {s['scheme']} | {s['beta']:.9g} | " +
                      ' | '.join(f"{r['clean_cosine_min']:.6f} / {r['clean_cosine_median']:.6f}" for r in selected_rows) + ' |')
    report += ['', '**With sigma=1e-3: no matching initial noisy-gradient cosine measurements were located in the '
               'existing evidence.** The training accuracy or scalar gradient traces cannot supply those missing BPTT/EP angles. '
               'Clean cosine passing does not imply noisy cosine passing.', '',
               '## B. Initial state RMS displacement', '',
               'Below: pooled RMS((v_plus-v_minus)/2), in simulator voltage units, across all elements in all 36 batches. '
               'This centered half-difference separates the odd nudge signal from common continued relaxation. '
               'The CSV also supplies both RMS(v_plus-v_free) and RMS(v_minus-v_free).', '',
               '| Threshold | Scheme | Hidden 1 | Hidden 2 | Hidden 3 | Output |',
               '|---|---|---:|---:|---:|---:|']
    for s in selections:
        selected_rows = [r for r in rows if r['threshold'] == s['threshold'] and r['scheme'] == s['scheme']]
        report.append(f"| {s['threshold']:.2f} | {s['scheme']} | " +
                      ' | '.join(f"{r['physical_centered_half_difference_rms']:.5g}" for r in selected_rows) + ' |')
    report += ['', 'For the literal free-to-positive-nudged displacement, pooled RMS(v_plus-v_free):', '',
               '| Threshold | Scheme | Hidden 1 | Hidden 2 | Hidden 3 | Output |',
               '|---|---|---:|---:|---:|---:|']
    for s in selections:
        selected_rows = [r for r in rows if r['threshold'] == s['threshold'] and r['scheme'] == s['scheme']]
        report.append(f"| {s['threshold']:.2f} | {s['scheme']} | " +
                      ' | '.join(f"{r['physical_positive_free_rms']:.5g}" for r in selected_rows) + ' |')
    report += ['', 'At identical initial parameters and inputs these **physical displacements are the same with and '
               'without read noise by construction**: noise is added only to copied endpoint voltages for gradient '
               'readout, after clean relaxation. This does not claim equal states later along noisy training.', '',
               'The apparent displacement of the noisy readings differs. For independent endpoint noise with sigma=1e-3, '
               'sqrt(E[RMS_squared]) for the centered half-difference is sqrt(d_odd^2 + sigma^2/2), giving a '
               '7.071e-4 noise floor. For one noisy positive endpoint against the exact free state it is '
               'sqrt(d_plus^2 + sigma^2). These are analytic expectations, not measured noisy displacements; '
               'they are included separately in the CSV. No sampled noisy displacement or cosine is fabricated.', '',
               '## C. Existing accuracy', '',
               '| Threshold | Scheme | Beta | Clean validation | sigma=1e-3 validation/outcome |',
               '|---|---|---:|---|---|']
    for r in accuracies:
        clean = f"{r['clean_validation_pct']:.2f}% (epoch {r['clean_epochs']})" if r['clean_validation_pct'] is not None else 'Not measured'
        noisy = (f"{r['noisy_final_validation_pct']:.2f}% (epoch 30)" if r['noisy_final_validation_pct'] is not None
                 else f"Nonfinite in epoch {r['noisy_epochs_completed'] + 1}; last {r['noisy_last_validation_pct']:.2f}% at epoch {r['noisy_epochs_completed']}"
                 if r['noisy_state'] == 'failed' else 'Not measured')
        report.append(f"| {r['threshold']:.2f} | {r['scheme']} | {r['beta']:.9g} | {clean} | {noisy} |")
    report += ['', 'Additional exact-beta evidence: ours beta0.9 (the original-grid p99 choice) completed ten clean '
               'epochs at 98.46%; it is not a training result for beta0.987334 and is not a 30-epoch comparison. '
               'The p90 clean/noisy pairs above use RTX5090 controls. Baseline p99 clean is also RTX5090. '
               'All are single-seed observations; different scheme-specific learning rates and beta-grid resolution '
               'limit causal comparisons.', '',
               'The p90 hidden-layer phase signal is larger than for the p99 choices, but the first two hidden-layer '
               'signals remain below the sigma/sqrt(2) readout floor. This voltage-level comparison does not determine '
               'gradient cosine or accuracy, because gradients aggregate products across sites and samples.', '',
               f'[Layer CSV]({STEM}_layers.csv) · [Accuracy CSV]({STEM}_accuracy.csv) · '
               f'[Source validation and hashes]({STEM}_provenance.json)', '',
               'Source studies: [calibration](beta_rule_comparison_20260918.md), '
               '[refinement](beta_refinement_20260919.md), '
               '[p90 noise extension](conv3_p90_read_noise_1em3_20260920.md), '
               '[historical baseline](baseline_read_noise_results_20260916.md), '
               '[beta0.9 ten-epoch training](layerwise_beta_training_20260919.md).', '']
    (OUT / f'{STEM}.md').write_text('\n'.join(report))
    (OUT / f'{STEM}_provenance.json').write_text(json.dumps(dict(
        new_simulations=0, selections=selections, initialization_cohort=cohort,
        calibration_bundles=provenance, accuracy=accuracies,
        missing_noisy_initial_cosines=True, official_test_read=False,
        script_sha256=sha256_file(Path(__file__))), indent=2) + '\n')
    print('\n'.join(report))


if __name__ == '__main__':
    main()
