"""Validate and summarize an initialized Conv3 output-displacement replay."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

from experiments.reporting import sha256_file, validate_run


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--config', default='config-retry.json')
    parser.add_argument('--analysis-dir', type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    config_path = workspace / args.config
    cfg = json.loads(config_path.read_text())
    output = workspace / cfg['output_root']
    execution = json.loads((output / 'summary.json').read_text())
    smoke = json.loads((output / 'smoke_summary.json').read_text())
    assert execution['state'] == smoke['state'] == 'complete'
    assert len(execution['cases']) == len(smoke['cases']) == len(cfg['cases'])
    assert execution['config_sha256'] == smoke['config_sha256'] == sha256_file(config_path)
    assert execution['runner_sha256'] == smoke['runner_sha256'] == sha256_file(
        workspace / 'experiments/replay_conv3_init_beta_noise.py')
    assert execution['cohort_sha256'] == cfg['expected_cohort_sha256']
    assert (output / 'exit_code').read_text().strip() == '0'
    target = cfg['target_output_rms']
    assert target > 0
    grouped = defaultdict(list)
    sources = []
    residual_failures = residual_count = 0
    for declared in cfg['cases']:
        for kind in ('smoke', 'runs'):
            run = output / kind / declared['name']
            assert not validate_run(run), run
            result = json.loads((run / 'result.json').read_text())
            metrics = result['terminal_metrics']
            assert metrics['replay_batches'] == (1 if kind == 'smoke' else cfg['expected_batches'])
            assert metrics['checkpoint_epoch'] == 0
            assert metrics['source_bytes_unchanged'] and metrics['all_parameter_tensors_unchanged']
            assert metrics['all_bias_tensors_exact_zero']
            assert not any(metrics[k] for k in ('official_test_read', 'optimizer_steps_applied', 'accuracy_evaluation'))
            assert sha256_file(run / 'checkpoints/initialization.pt') == cfg['initializer_checkpoint_sha256']
            if kind == 'smoke':
                continue
            signals = read_csv(run / 'state_signal.csv')
            assert len(signals) == cfg['expected_batches'] * 4
            assert {(int(r['batch_index']), int(r['state_layer_index'])) for r in signals} == {
                (b, layer) for b in range(cfg['expected_batches']) for layer in range(4)}
            for row in signals:
                assert row['scheme'] == declared['scheme']
                assert float(row['injected_beta']) == declared['injected_beta']
                grouped[declared['scheme'], declared['injected_beta'], int(row['state_layer_index'])].append(row)
            residuals = read_csv(run / 'equilibrium_residuals.csv')
            residual_count += len(residuals)
            residual_failures += sum(r['gate_passed'].lower() != 'true' for r in residuals)
            sources.append({'case': declared['name'], 'result_sha256': sha256_file(run / 'result.json')})

    summaries = []
    for (scheme, beta, layer), rows in grouped.items():
        count = sum(int(r['element_count']) for r in rows)
        summary = dict(scheme=scheme, injected_beta=beta, layer=layer + 1,
                       state_layer=rows[0]['state_layer_name'], batches=len(rows), element_count=count)
        for field in ('free_rms', 'positive_free_rms', 'negative_free_rms', 'centered_half_difference_rms'):
            value = math.sqrt(sum(int(r['element_count']) * float(r[field])**2 for r in rows) / count)
            assert math.isfinite(value)
            summary[field] = value
        summary['positive_target_difference'] = summary['positive_free_rms'] - target
        summary['positive_target_error_percent'] = 100 * summary['positive_target_difference'] / target
        summary['negative_target_error_percent'] = 100 * (summary['negative_free_rms'] / target - 1)
        summary['positive_batch_rms_min'] = min(float(r['positive_free_rms']) for r in rows)
        summary['positive_batch_rms_max'] = max(float(r['positive_free_rms']) for r in rows)
        summaries.append(summary)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    with (args.analysis_dir / 'layer_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    outputs = [r for r in summaries if r['layer'] == 4]
    assert len(outputs) == 3 and len(summaries) == 12
    validation = dict(state='complete', complete_cases=3, validated_bundles=6,
                      production_batches=108, state_signal_rows=432,
                      cohort_sha256=execution['cohort_sha256'],
                      source_checkpoint_sha256=cfg['initializer_checkpoint_sha256'],
                      residual_rows=residual_count, residual_failures=residual_failures,
                      sources=sources, elapsed_seconds=execution['total_including_smoke_seconds'],
                      outputs=outputs, official_test_read=False, optimizer_steps_applied=False,
                      excluded_attempts=['original Akib smoke: GPU OOM; no production launched'])
    (args.analysis_dir / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    lines = ['# Conv3 initialization: output displacement target of one', '',
             'Measured on the shared saved seed-0 initializer, 36 identical MNIST validation batches '
             'of 16, T=K=8, float64, frozen-current centered EqProp, wide [0,100] weights, '
             'zero biases and clean endpoints. RMS is pooled over all examples and output coordinates.', '',
             '| Scheme | Injected beta | Positive-free RMS | Negative-free RMS | Centered half-difference RMS | Positive deviation from 1 |',
             '|---|---:|---:|---:|---:|---:|']
    for r in outputs:
        lines.append(f"| {r['scheme']} | {r['injected_beta']:.8g} | {r['positive_free_rms']:.10f} | "
                     f"{r['negative_free_rms']:.10f} | {r['centered_half_difference_rms']:.10f} | "
                     f"{r['positive_target_error_percent']:+.6f}% |")
    lines += ['', 'All three scientific cases and three one-batch smokes validate locally. '
              'The initializer bytes, in-memory weights, zero biases and exact prior cohort are preserved. '
              f'Projected-KKT residual failures: {residual_failures}/{residual_count}.', '',
              'Akib RTX3080. The first smoke ran out of GPU memory; its failed bundle is retained. '
              'The successful attempt used expandable allocator segments and a 256-MiB cuDNN workspace cap, '
              'with the scientific configuration unchanged. '
              f"Successful smoke plus full replay took {execution['total_including_smoke_seconds']:.2f} seconds.", '',
              'This verifies the pooled initial output displacement for one initializer. It does not '
              'establish equal hidden-layer responses, per-example displacement of one, or training stability. '
              'No optimizer step or official-test access occurred.', '',
              '[Per-layer measurements and batch RMS ranges](layer_summary.csv) · '
              '[Coverage and provenance](validation.json)']
    (args.analysis_dir / 'report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
