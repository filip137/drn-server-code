"""Read-only final-checkpoint EP/BPTT gradient comparison for the p90 noise sweep."""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/eqprop-conv3-p90-final-noise-cosine-20260921-v1'


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    spec = importlib.util.spec_from_file_location('final_cosine_gate', OUT / 'source/experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    base, shadow, torch = gate.base, gate.shadow, gate.torch
    from experiments.reporting import start_run, complete_run, fail_run, append_metric, update_status_progress, validate_run
    for name, expected in cfg['analyzer_sha256'].items():
        assert base.sha256_file(OUT / 'source/experiments' / name) == expected, name
    source = Path(cfg['source_contract']['runtime_source_root'])
    for name, expected in cfg['runtime_sha256'].items():
        assert base.sha256_file(source / name) == expected, name
    cases = cfg['cases']
    if args.smoke:
        # One clean and one noisy checkpoint exercise exact final loading and noise readout.
        cases = [next(c for c in cases if c['sigma'] == 0 and c['scheme'] == 'baseline'),
                 next(c for c in cases if c['sigma'] > 0 and c['scheme'] == 'legacy')]
    started = time.monotonic()
    dataset_cfg = json.loads((ROOT / cases[0]['bundle'] / 'config.used.json').read_text())
    batches, cohort = gate._build_cohort(cfg, [{'source_config': dataset_cfg}])
    if args.smoke:
        batches = batches[:1]
    (OUT / ('smoke_cohort.json' if args.smoke else 'cohort.json')).write_text(json.dumps(cohort, indent=2) + '\n')
    summary = []
    for declared in cases:
        if time.monotonic() - started > cfg['deadline_gpu_seconds']:
            raise TimeoutError('Declared replay budget exhausted.')
        bundle = ROOT / declared['bundle']
        assert not validate_run(bundle), bundle
        for name, expected in declared['source_hashes'].items():
            assert base.sha256_file(bundle / name) == expected, name
        source_config = json.loads((bundle / 'config.used.json').read_text())
        assert source_config['training_algorithm'] == 'EP' and source_config['runtime_dtype'] == 'float64'
        assert math.isclose(source_config['eqprop']['injected_beta_B'], declared['beta'], rel_tol=1e-12)
        assert float(source_config['eqprop']['endpoint_read_noise_std']) == declared['sigma']
        gate._bias_lr_proof(source_config)
        model = base._model_config(source_config)
        assert model['num_iterations_inference'] == model['num_iterations_training'] == 8
        voltage, current = float(model['voltage_amp']), float(model['current_amp'])
        case = dict(architecture='conv3', scheme=declared['scheme'], T=8, K=8, native_T=8, native_K=8,
                    base_beta=declared['beta'] / (voltage / current)**3, injected_beta=declared['beta'],
                    voltage_amp=voltage, current_amp=current, voltage_amplification=voltage,
                    current_amplification=current, source_config=source_config, native_checkpoint_dtype='float64',
                    final_checkpoint_path=bundle / 'final_model.pt', weights_final_path=bundle / 'weights_final.npz',
                    final_checkpoint_sha256=declared['source_hashes']['final_model.pt'])
        selected = gate._selected(case, 'final')
        run = OUT / ('smoke' if args.smoke else 'runs') / declared['name']
        manifest = dict(study_id=cfg['study_id'], run_id=declared['name'], arm_id=declared['name'],
                        evidence_class='ordinary_mnist_selection', dataset='ordinary_mnist_validation', smoke=args.smoke,
                        command=[sys.executable, *sys.argv], configuration=dict(path=str(args.config),
                        sha256=base.sha256_file(args.config), resolved=cfg), inputs=declared,
                        git=dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                                 runner_sha256=base.sha256_file(Path(__file__))),
                        runtime=dict(base.runtime_context(target='local:RTX3090'), torch_version=torch.__version__,
                                     cuda_version=torch.version.cuda, device=args.device),
                        checkpoint_role='final', checkpoint_epoch=30, cohort_sha256=cohort['cohort_sha256'],
                        official_test_read=False, optimizer_steps_applied=False)
        start_run(run, manifest)
        rows, guards, residuals = [], [], []
        case_started = time.monotonic()
        try:
            for batch in batches:
                if time.monotonic() - started > cfg['deadline_gpu_seconds']:
                    raise TimeoutError('Declared replay budget exhausted.')
                output = shadow._run_precision(case=case, selected=selected, batch=batch,
                    precision='float64', device=torch.device(args.device), residual_threshold=.01,
                    amplification_exponent=3, nudging_mode='current', eqprop_variant='centered',
                    capture_endpoint_states=True)
                assert output['guard']['parameter_tensors_unchanged_after_cast']
                runtime = output['captured_runtime']
                assert all(bool((p.state == 0).all()) for p in runtime['parameters'] if str(p.name).startswith('Bias_'))
                residual_pass = all(bool(r['gate_passed']) for r in output['residual_rows'])
                residuals.extend(output['residual_rows'])
                guards.append(output['guard'])
                draws = [(0.0, None)]
                if declared['sigma'] > 0:
                    draws += [(declared['sigma'], seed) for seed in cfg['read_noise_seeds']]
                for sigma, seed in draws:
                    noise = gate._read_noise_contract({})
                    noise.update(endpoint_read_noise_std=sigma, endpoint_read_noise_seed=seed)
                    scored, numerators, noise_rows, noise_guard = gate._apply_endpoint_read_noise(
                        output=output, case=case, batch=batch, noise_contract=noise, device=torch.device(args.device))
                    output.update(scored_gradients=scored, scored_numerators=numerators, noise_contract=noise)
                    assert noise_guard['parameter_tensors_unchanged']
                    for parameter in output['gradients']:
                        row = gate._layer_row(config=cfg, case=case, selected=selected, batch=batch,
                                              parameter_name=parameter, output=output, residual_passed=residual_pass)
                        row.update(training_sigma=declared['sigma'], training_gpu=declared['gpu'],
                                   checkpoint_epoch=30, noise_draw_seed=seed,
                                   checkpoint_sha256=case['final_checkpoint_sha256'], source_bundle=declared['bundle'])
                        row['bptt_rms'] = row['bptt_l2'] / math.sqrt(row['element_count'])
                        row['eqprop_rms'] = row['eqprop_l2'] / math.sqrt(row['element_count'])
                        rows.append(row)
                append_metric(run / 'metrics.jsonl', dict(stage='checkpoint_replay', epoch=30,
                              dataset_split='validation', batch_index=int(batch['batch_index']), comparisons=len(rows)))
                update_status_progress(run, dict(stage='checkpoint_replay', batch_index=int(batch['batch_index']), batches=len(batches)))
            for name, expected in declared['source_hashes'].items():
                assert base.sha256_file(bundle / name) == expected, name
            assert len(rows) == len(batches) * 4 * (5 if declared['sigma'] > 0 else 1)
            write_csv(run / 'layer_metrics.csv', rows)
            write_csv(run / 'equilibrium_residuals.csv', residuals)
            (run / 'read_only_guards.json').write_text(json.dumps(guards, indent=2) + '\n')
            complete_run(run, terminal_metrics=dict(replay_batches=len(batches), layer_comparisons=len(rows),
                         source_bytes_unchanged=True, all_parameter_tensors_unchanged=True,
                         checkpoint_epoch=30, checkpoint_role='final', official_test_read=False,
                         optimizer_steps_applied=False, elapsed_seconds=time.monotonic()-case_started),
                         completion=dict(criteria_met=True, no_undefined_cosines=all(r['cosine'] is not None for r in rows)))
            assert not validate_run(run)
            summary.append(dict(name=declared['name'], state='complete', bundle=str(run.relative_to(ROOT)),
                                seconds=time.monotonic()-case_started))
            (OUT / ('smoke_summary.json' if args.smoke else 'summary.json')).write_text(json.dumps(summary, indent=2) + '\n')
            print('DONE', declared['name'], round(time.monotonic()-case_started, 1), flush=True)
        except BaseException as exc:
            if rows:
                write_csv(run / 'partial_layer_metrics.csv', rows)
            fail_run(run, error=exc)
            raise
    print('COMPLETE', len(summary), 'checkpoints', round(time.monotonic()-started, 1), 'seconds', flush=True)


if __name__ == '__main__':
    main()
