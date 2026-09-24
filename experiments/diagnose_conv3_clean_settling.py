"""Bounded clean-only settling check for one trained-checkpoint replay failure."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', default='A100_ours_sigma0.0005')
    parser.add_argument('--inference-iterations', type=int, nargs='+', default=[8, 32, 128, 512])
    parser.add_argument('--batches', type=int, default=3)
    parser.add_argument('--budget-seconds', type=float, default=600)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = ROOT / cfg['output_root'] / 'settling_diagnostic'
    assert cfg['study_id'] in (ROOT / 'docs/current_simulations.md').read_text()
    spec = importlib.util.spec_from_file_location('settling_gate', ROOT / cfg['analyzer_source_root'] /
        'analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    base, shadow, torch = gate.base, gate.shadow, gate.torch
    from experiments.reporting import append_metric, atomic_write_json, complete_run, fail_run, start_run, validate_run

    prior_path = ROOT / cfg['cohort_config']
    assert base.sha256_file(prior_path) == cfg['cohort_config_sha256']
    prior = json.loads(prior_path.read_text())
    for name, digest in cfg['analyzer_sha256'].items():
        assert base.sha256_file(ROOT / cfg['analyzer_source_root'] / name) == digest
    for name, digest in prior['runtime_sha256'].items():
        assert base.sha256_file(ROOT / cfg['source_contract']['runtime_source_root'] / name) == digest
    declared = next(c for c in cfg['cases'] if c['name'] == args.checkpoint)
    bundle = ROOT / declared['bundle']
    for name, digest in declared['source_hashes'].items():
        assert base.sha256_file(bundle / name) == digest
    source = json.loads((bundle / 'config.used.json').read_text())
    model = base._model_config(source)
    resolved = {**prior, **cfg, 'dataset': dict(prior['dataset'], root=cfg['data_root'])}
    batches, cohort = gate._build_cohort(resolved, [{'source_config': source}])
    batches = batches[:args.batches]
    assert len(batches) == args.batches and args.batches > 0
    atomic_write_json(out / 'cohort.json', dict(cohort, replayed_batch_indices=[int(b['batch_index']) for b in batches]))
    voltage, current = float(model['voltage_amp']), float(model['current_amp'])
    beta = declared['beta'] * min(cfg['beta_factors'])
    started = time.monotonic()
    for inference in args.inference_iterations:
        case = dict(architecture='conv3', scheme=declared['scheme'], T=inference, K=8, native_T=8, native_K=8,
            base_beta=beta/(voltage/current)**3, injected_beta=beta, voltage_amp=voltage, current_amp=current,
            voltage_amplification=voltage, current_amplification=current, source_config=source,
            native_checkpoint_dtype='float64', final_checkpoint_path=bundle/'final_model.pt',
            weights_final_path=bundle/'weights_final.npz', final_checkpoint_sha256=declared['source_hashes']['final_model.pt'])
        selected = gate._selected(case, 'final')
        run = out / f'T{inference}_K8'
        if run.exists():
            raise FileExistsError(run)
        start_run(run, dict(study_id=cfg['study_id'], run_id=run.name, arm_id=run.name,
            evidence_class='ordinary_mnist_selection', evidence_tier='exploratory',
            command=[sys.executable, *sys.argv], inputs=declared,
            configuration=dict(path=str(args.config), sha256=base.sha256_file(args.config), resolved=cfg,
                diagnostic_overrides=dict(T=inference, K=8, injected_beta=beta, read_noise_std=0, batches=args.batches)),
            git=dict(commit=cfg['source_commit'], runner_sha256=base.sha256_file(Path(__file__)), dirty_worktree=True),
            runtime=dict(base.runtime_context(target=cfg['target']), torch_version=torch.__version__, cuda_version=torch.version.cuda),
            main_sweep_member=False, official_test_read=False, optimizer_steps_applied=False, accuracy_evaluation=False))
        rows, residuals, guards = [], [], []
        try:
            for batch in batches:
                if time.monotonic()-started >= args.budget_seconds:
                    raise TimeoutError('Clean settling diagnostic budget exhausted')
                output = shadow._run_precision(case=case, selected=selected, batch=batch, precision='float64',
                    device=torch.device('cuda'), residual_threshold=.01, amplification_exponent=3,
                    nudging_mode='current', eqprop_variant='centered')
                assert output['guard']['parameter_tensors_unchanged_after_cast']
                assert output['guard']['frozen_current_force_matches_post_T_cost_gradient']
                guards.append(output['guard'])
                residuals.extend(output['residual_rows'])
                for name in output['gradients']:
                    rows.append(gate._layer_row(config=resolved, case=case, selected=selected, batch=batch,
                        parameter_name=name, output=output,
                        residual_passed=all(r['gate_passed'] for r in output['residual_rows'])))
                append_metric(run/'metrics.jsonl', dict(stage='clean_settling_diagnostic',
                    batch_index=int(batch['batch_index']), comparisons=len(rows)))
                print('T', inference, 'K', 8, 'batch', batch['batch_index'],
                      'cosines', [round(r['cosine'], 6) for r in rows[-4:]], flush=True)
                del output
                gc.collect()
            base._write_csv(run/'layer_metrics.csv', rows)
            base._write_csv(run/'equilibrium_residuals.csv', residuals)
            atomic_write_json(run/'read_only_guards.json', guards)
            for name, digest in declared['source_hashes'].items():
                assert base.sha256_file(bundle/name) == digest
            complete_run(run, terminal_metrics=dict(replay_batches=len(batches), layer_comparisons=len(rows),
                residual_failed_rows=sum(not r['gate_passed'] for r in residuals), source_bytes_unchanged=True,
                all_parameter_tensors_unchanged=True, official_test_read=False, optimizer_steps_applied=False),
                completion=dict(criteria_met=True, coverage_complete=True))
            assert not validate_run(run)
        except BaseException as exc:
            fail_run(run, error=exc)
            raise
    atomic_write_json(out/'summary.json', dict(state='complete', elapsed_seconds=time.monotonic()-started,
        checkpoint=args.checkpoint, inference_iterations=args.inference_iterations, K=8, replay_batches=args.batches,
        injected_beta=beta, read_noise_std=0, main_sweep_member=False))


if __name__ == '__main__':
    main()
