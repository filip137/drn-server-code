"""Find the first residual-qualified integer T for a saved Conv3 checkpoint."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    parent = json.loads((ROOT / cfg['replay_config']).read_text())
    assert cfg['study_id'] in (ROOT / 'docs/current_simulations.md').read_text()
    spec = importlib.util.spec_from_file_location('minimum_t_gate', ROOT / parent['analyzer_source_root'] /
        'analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    base, shadow, torch = gate.base, gate.shadow, gate.torch
    from experiments.reporting import append_metric, atomic_write_json, complete_run, fail_run, start_run, update_status_progress, validate_run

    assert base.sha256_file(ROOT / cfg['replay_config']) == cfg['replay_config_sha256']
    prior_path = ROOT / parent['cohort_config']
    assert base.sha256_file(prior_path) == parent['cohort_config_sha256']
    prior = json.loads(prior_path.read_text())
    for name, digest in parent['analyzer_sha256'].items():
        assert base.sha256_file(ROOT / parent['analyzer_source_root'] / name) == digest
    for name, digest in prior['runtime_sha256'].items():
        assert base.sha256_file(ROOT / parent['source_contract']['runtime_source_root'] / name) == digest
    declared = next(c for c in parent['cases'] if c['name'] == cfg['checkpoint'])
    bundle = ROOT / declared['bundle']
    def verify_source():
        for name, digest in declared['source_hashes'].items():
            assert base.sha256_file(bundle / name) == digest
    verify_source()
    source = json.loads((bundle / 'config.used.json').read_text())
    model = base._model_config(source)
    resolved = {**prior, **parent, 'dataset': dict(prior['dataset'], root=parent['data_root'])}
    batches, cohort = gate._build_cohort(resolved, [{'source_config': source}])
    assert len(batches) == cfg['expected_batches'] == 36
    if args.smoke:
        batches = batches[:1]
    out = ROOT / cfg['output_root'] / ('smoke' if args.smoke else 'production')
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    runner_hash, config_hash = base.sha256_file(Path(__file__)), base.sha256_file(args.config)
    smoke_seconds = 0.
    if not args.smoke:
        smoke = json.loads((out.parent / 'smoke/summary.json').read_text())
        assert smoke['state'] == 'complete' and smoke['runner_sha256'] == runner_hash
        assert smoke['config_sha256'] == config_hash
        assert smoke['T64_free_gate_passed'] and smoke['minimum_complete_phase_T'] is not None
        smoke_seconds = smoke['elapsed_seconds']
    def check_deadline():
        if time.monotonic() - started + smoke_seconds >= cfg['budget_seconds']:
            raise TimeoutError('Minimum-T diagnostic budget exhausted')
    atomic_write_json(out / 'cohort.json', dict(cohort, replayed_batch_indices=[int(b['batch_index']) for b in batches]))
    voltage, current = float(model['voltage_amp']), float(model['current_amp'])
    beta = declared['beta'] * cfg['beta_factor']
    def case_at(t):
        return dict(architecture='conv3', scheme=declared['scheme'], T=t, K=cfg['K'], native_T=8, native_K=8,
            base_beta=beta/(voltage/current)**3, injected_beta=beta, voltage_amp=voltage, current_amp=current,
            voltage_amplification=voltage, current_amplification=current, source_config=source,
            native_checkpoint_dtype='float64', final_checkpoint_path=bundle/'final_model.pt',
            weights_final_path=bundle/'weights_final.npz', final_checkpoint_sha256=declared['source_hashes']['final_model.pt'])
    def begin(name, settings):
        run = out / name
        start_run(run, dict(study_id=cfg['study_id'], run_id=name, arm_id=name,
            evidence_class='ordinary_mnist_selection', evidence_tier='exploratory', smoke=args.smoke,
            command=[sys.executable, *sys.argv], inputs=declared,
            configuration=dict(path=str(args.config), sha256=config_hash, resolved=cfg, measurement=settings),
            git=dict(commit=parent['source_commit'], runner_sha256=runner_hash, dirty_worktree=True),
            runtime=dict(base.runtime_context(target=cfg['target']), torch_version=torch.__version__, cuda_version=torch.version.cuda),
            official_test_read=False, optimizer_steps_applied=False, accuracy_evaluation=False))
        return run

    old = ROOT / parent['output_root']
    t8_guards = json.loads((old / 'runs' / (cfg['checkpoint'] + '_factor_0p0001') / 'read_only_guards.json').read_text())
    t32_guards = json.loads((old / 'settling_diagnostic/T32_K8/read_only_guards.json').read_text())
    raw = np.empty((len(batches), cfg['T_max'], 4, cfg['batch_size']), dtype=np.float64)
    run = begin('free_trace', dict(T_range=[1, cfg['T_max']], phase='free'))
    state_hash_checks = []
    try:
        for bi, batch in enumerate(batches):
            check_deadline()
            runtime, guard = shadow._checkpoint_runtime(case_at(8), role='final', device=torch.device('cuda'),
                gradient_iterations=cfg['K'], nudging_mode='current')
            shadow._convert_runtime_dtype(runtime, torch.float64)
            before = base._parameter_state_sha256(runtime['parameters'])
            shadow._set_input_and_reset(runtime, batch['images'].to(device='cuda', dtype=torch.float64), dtype=torch.float64)
            runtime['minimizer_inference'].num_iterations = 1
            for t in range(1, cfg['T_max'] + 1):
                runtime['minimizer_inference'].compute_equilibrium()
                residual = shadow.extended._ResidualAccumulator()
                residual.add_endpoint(runtime, gradient_source=runtime['energy_fn'], phase='post_T_free', beta=0.,
                    source_indices=batch['source_indices'])
                for key, record in residual.records.items():
                    raw[bi, t-1, key[2], :] = record['selected_max']
                if t == 8 or (t == 32 and bi < len(t32_guards)):
                    expected = (t8_guards if t == 8 else t32_guards)[bi]['common_post_T_state_sha256']
                    actual = base._layer_state_sha256(runtime['free_layers'])
                    assert actual == expected, ('Sequential trace differs from direct minimization', bi, t)
                    state_hash_checks.append(dict(batch_index=bi, T=t, state_sha256=actual, exact_match=True))
            assert base._parameter_state_sha256(runtime['parameters']) == before
            append_metric(run/'metrics.jsonl', dict(stage='free_residual_trace', batches_complete=bi+1, T_max=cfg['T_max']))
            update_status_progress(run, dict(stage='free_residual_trace', batches_complete=bi+1, batches=len(batches)))
            print('TRACE', bi+1, '/', len(batches), flush=True)
            del runtime
            gc.collect()
        assert np.isfinite(raw).all()
        batch_p90 = np.quantile(raw, .9, axis=-1)
        batch_maximum = raw.max(axis=-1)
        pooled = raw.transpose(1, 2, 0, 3).reshape(cfg['T_max'], 4, -1)
        pooled_p90 = np.quantile(pooled, .9, axis=-1)
        threshold, hard = cfg['p90_threshold'], cfg['hard_maximum_threshold']
        strict_pass = ((batch_p90 < threshold) & (batch_maximum < hard)).all(axis=(0, 2))
        pooled_pass = ((pooled_p90 < threshold) & (pooled.max(axis=-1) < hard)).all(axis=1)
        uniform_pass = (pooled.max(axis=-1) < threshold).all(axis=1)
        def first(mask):
            indices = np.flatnonzero(mask)
            return int(indices[0])+1 if len(indices) else None
        trace_rows = []
        for ti in range(cfg['T_max']):
            row = dict(T=ti+1, all_batch_gates_passed=bool(strict_pass[ti]),
                pooled_gate_passed=bool(pooled_pass[ti]), uniform_maximum_gate_passed=bool(uniform_pass[ti]),
                worst_batch_p90=float(batch_p90[:, ti, :].max()), maximum_residual=float(pooled[ti].max()),
                failed_batch_layer_rows=int(((batch_p90[:, ti, :] >= threshold) | (batch_maximum[:, ti, :] >= hard)).sum()))
            for li in range(4):
                row[f'layer_{li+1}_pooled_p90'] = float(pooled_p90[ti, li])
            trace_rows.append(row)
        np.savez_compressed(run/'per_example_residuals.npz', projected_or_raw_maxima=raw,
            source_indices=np.asarray([b['source_indices'] for b in batches], dtype=np.int64))
        base._write_csv(run/'residual_vs_T.csv', trace_rows)
        atomic_write_json(run/'direct_state_equivalence.json', state_hash_checks)
        verify_source()
        trace_result = dict(minimum_all_batch_T=first(strict_pass), minimum_pooled_T=first(pooled_pass),
            minimum_uniform_T=first(uniform_pass), T64_free_gate_passed=bool(strict_pass[-1]),
            replay_batches=len(batches), exact_direct_state_checks=len(state_hash_checks),
            source_bytes_unchanged=True, all_parameter_tensors_unchanged=True)
        complete_run(run, terminal_metrics=trace_result, completion=dict(criteria_met=True, coverage_complete=True))
        assert not validate_run(run)
        print('TRACE_COMPLETE', json.dumps(trace_result), flush=True)
    except BaseException as exc:
        fail_run(run, error=exc)
        raise

    phase_results = {}
    def replay(t):
        check_deadline()
        case = case_at(t)
        selected = gate._selected(case, 'final')
        run = begin(f'clean_T{t}_K{cfg["K"]}', dict(T=t, K=cfg['K'], injected_beta=beta, read_noise_std=0))
        rows, residuals, guards = [], [], []
        try:
            for bi, batch in enumerate(batches):
                check_deadline()
                output = shadow._run_precision(case=case, selected=selected, batch=batch, precision='float64',
                    device=torch.device('cuda'), residual_threshold=threshold, amplification_exponent=3,
                    nudging_mode='current', eqprop_variant='centered')
                assert output['guard']['parameter_tensors_unchanged_after_cast']
                assert output['guard']['frozen_current_force_matches_post_T_cost_gradient']
                guards.append(output['guard'])
                residuals.extend(output['residual_rows'])
                for r in output['residual_rows']:
                    if r['phase'] == 'post_T_free':
                        assert np.isclose(r['selected_max_p90'], batch_p90[bi, t-1, r['layer_index']], rtol=1e-12, atol=1e-15)
                for name in output['gradients']:
                    rows.append(gate._layer_row(config=resolved, case=case, selected=selected, batch=batch,
                        parameter_name=name, output=output, residual_passed=all(r['gate_passed'] for r in output['residual_rows'])))
                append_metric(run/'metrics.jsonl', dict(stage='full_clean_replay', batch_index=bi, T=t, comparisons=len(rows)))
                update_status_progress(run, dict(stage='full_clean_replay', batches_complete=bi+1, batches=len(batches), T=t))
                print('REPLAY', t, bi+1, '/', len(batches), flush=True)
                del output
                gc.collect()
            base._write_csv(run/'layer_metrics.csv', rows)
            base._write_csv(run/'equilibrium_residuals.csv', residuals)
            atomic_write_json(run/'read_only_guards.json', guards)
            verify_source()
            result = dict(T=t, replay_batches=len(batches), all_phase_gates_passed=all(r['gate_passed'] for r in residuals),
                residual_failed_rows=sum(not r['gate_passed'] for r in residuals),
                minimum_individual_cosine=min(r['cosine'] for r in rows),
                source_bytes_unchanged=True, all_parameter_tensors_unchanged=True)
            complete_run(run, terminal_metrics=result, completion=dict(criteria_met=True, coverage_complete=True))
            assert not validate_run(run)
            phase_results[t] = result
            print('REPLAY_COMPLETE', json.dumps(result), flush=True)
            return result
        except BaseException as exc:
            fail_run(run, error=exc)
            raise
    minimum_complete = None
    if strict_pass[-1]:
        for ti in np.flatnonzero(strict_pass):
            candidate = int(ti)+1
            if replay(candidate)['all_phase_gates_passed']:
                minimum_complete = candidate
                break
        for t in cfg['reference_T']:
            if t not in phase_results:
                replay(t)
    summary = dict(state='complete', **trace_result, minimum_complete_phase_T=minimum_complete,
        phase_results=phase_results, config_sha256=config_hash, runner_sha256=runner_hash,
        elapsed_seconds=time.monotonic()-started, total_including_smoke_seconds=time.monotonic()-started+smoke_seconds,
        official_test_read=False, optimizer_steps_applied=False, accuracy_evaluation=False)
    atomic_write_json(out/'summary.json', summary)
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
