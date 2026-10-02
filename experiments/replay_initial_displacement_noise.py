"""Matched Conv1/2/3 initialization displacement and endpoint-noise replay."""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def noise_draws(cfg):
    seeds = cfg['read_noise_seeds']
    assert len(seeds) == len(set(seeds)) == 3
    assert min(abs(a-b) for a in seeds for b in seeds if a != b) > 36000
    return [(0., None, -1)] + [(sigma, seed, i) for sigma in cfg['noise_sigmas'][1:]
                              for i,seed in enumerate(seeds)]


def check_noise_record(seen, record, *, architecture, sigma, seed, batch_index):
    phase_index = {'negative':1, 'positive':2}[record['phase']]
    layer = record['state_layer_index']
    expected = seed + 1000*batch_index + 100*phase_index + layer
    assert record['seed'] == expected
    raw = ('raw',architecture,seed,batch_index,record['phase'],layer)
    identity = (expected,record['element_count'])
    assert seen.setdefault(raw,identity) == identity
    scaled = ('scaled',sigma,*raw[1:])
    digest = record['standard_normal_sha256']
    assert seen.setdefault(scaled,digest) == digest


class BpttCache:
    """Reuse the fixed reference only when parameters, input/cohort and post-T match."""
    def __init__(self, original, base, verify_cached=False):
        self.original, self.base = original, base
        self.verify_cached = verify_cached
        self.parity_checks = 0
        self.context = None
        self.values = {}
        self.computations = self.hits = 0

    def __call__(self, runtime, *, post_t_states, post_t_hash, expected_k):
        assert self.context is not None
        assert runtime['gradient_iterations'] == runtime['minimizer_gradient'].num_iterations == expected_k
        identity = (self.base._parameter_state_sha256(runtime['parameters']),post_t_hash,
                    expected_k,tuple(runtime['weight_names']))
        if self.context not in self.values:
            gradients,guard = self.original(runtime,post_t_states=post_t_states,
                post_t_hash=post_t_hash,expected_k=expected_k)
            self.values[self.context] = (identity,gradients,guard)
            self.computations += 1
        else:
            self.hits += 1
        saved,gradients,guard = self.values[self.context]
        assert saved == identity, 'BPTT cache input/state/parameter mismatch'
        if self.verify_cached and self.hits:
            fresh,fresh_guard = self.original(runtime,post_t_states=post_t_states,post_t_hash=post_t_hash,expected_k=expected_k)
            assert all(self.base._tensor_sha256(fresh[k]) == self.base._tensor_sha256(v) for k,v in gradients.items())
            assert fresh_guard == guard
            self.parity_checks += 1
        return {k:v.clone() for k,v in gradients.items()}, dict(guard,bptt_cached_reference=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--target')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = args.output_root.resolve() if args.output_root else ROOT / cfg['output_root']
    spec = importlib.util.spec_from_file_location(
        'init_noise_gate', ROOT / cfg['analyzer_source_root'] /
        'analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    base, shadow, torch = gate.base, gate.shadow, gate.torch
    from experiments.reporting import (
        append_metric, atomic_write_json, complete_run, fail_run,
        start_run, update_status_progress, validate_run,
    )

    def verify_inputs():
        for name, expected in cfg['analyzer_sha256'].items():
            assert base.sha256_file(ROOT / cfg['analyzer_source_root'] / name) == expected, name
        cohort_path = ROOT / cfg['cohort_config']
        assert base.sha256_file(cohort_path) == cfg['cohort_config_sha256']
        prior = json.loads(cohort_path.read_text())
        for name, expected in prior['runtime_sha256'].items():
            assert base.sha256_file(Path(cfg['source_contract']['runtime_source_root']) / name) == expected, name
        for case in cfg['cases']:
            assert base.sha256_file(ROOT / case['initializer_checkpoint_path']) == case['initializer_checkpoint_sha256']
            for reference in (case['calibration_proof'],):
                assert base.sha256_file(ROOT/reference['path']) == reference['sha256']
            assert base.sha256_file(ROOT / case['source_config']) == case['source_config_sha256']
        return prior

    prior = verify_inputs()
    resolved = {**cfg, 'dataset': copy.deepcopy(prior['dataset'])}
    if args.data_root:
        resolved['dataset']['root'] = str(args.data_root)
    assert len(cfg['cases']) == cfg['expected_case_count'] > 0
    assert len({c['name'] for c in cfg['cases']}) == len(cfg['cases'])
    assert cfg['noise_sigmas'] and cfg['noise_sigmas'][0] == 0.
    assert cfg['noise_sigmas'] == sorted(set(cfg['noise_sigmas']))
    assert all(math.isfinite(s) and s >= 0 for s in cfg['noise_sigmas'])
    draws = noise_draws(cfg)
    assert all(c['scheme'] in ('baseline', 'legacy', 'ours')
               and math.isfinite(c['injected_beta']) and c['injected_beta'] > 0
               for c in cfg['cases'])
    assert cfg['expected_layer_comparisons'] == (
        sum(int(c['architecture'][-1])+1 for c in cfg['cases']) * cfg['expected_batches'] * len(draws))
    for arch in {c['architecture'] for c in cfg['cases']}:
        assert len({c['reconstructed_initialization_tensor_sha256'] for c in cfg['cases'] if c['architecture']==arch}) == 1
    assert not cfg['official_test_read'] and not cfg['optimizer_steps_applied'] and not cfg['accuracy_evaluation']
    started = time.monotonic()
    smoke_summary_path = out / 'smoke_summary.json'
    if args.smoke:
        prior_seconds = 0.
    else:
        smoke = json.loads(smoke_summary_path.read_text())
        assert smoke['state'] == 'complete' and len(smoke['cases']) == len(cfg['cases'])
        assert smoke['config_sha256'] == base.sha256_file(args.config)
        assert smoke['runner_sha256'] == base.sha256_file(Path(__file__))
        for case in smoke['cases']:
            assert not validate_run(ROOT / case['run_dir'])
        prior_seconds = smoke['elapsed_seconds']

    def check_deadline():
        if time.monotonic() - started + prior_seconds >= cfg['budget_seconds']:
            raise TimeoutError('Declared replay budget exhausted.')

    dataset_source = json.loads((ROOT / cfg['cases'][0]['source_config']).read_text())
    batches, cohort = gate._build_cohort(resolved, [{'source_config': dataset_source}])
    assert len(batches) == cfg['expected_batches'] == 36
    assert cohort['cohort_sha256'] == cfg['expected_cohort_sha256']
    if args.smoke:
        batches = batches[:1]
    cohort_record = dict(cohort, replayed_batch_indices=[int(b['batch_index']) for b in batches])
    atomic_write_json(out / ('smoke_cohort.json' if args.smoke else 'cohort.json'), cohort_record)
    summary = []
    bptt_hashes, noise_hashes = {}, {}
    cache = BpttCache(shadow._native_bptt_weight_gradients,base,verify_cached=args.smoke)
    shadow._native_bptt_weight_gradients = cache
    for declared in cfg['cases']:
        check_deadline()
        source_config = json.loads((ROOT / declared['source_config']).read_text())
        gate._bias_lr_proof(source_config)
        source_config['init_checkpoint_path'] = str(ROOT / declared['initializer_checkpoint_path'])
        source_config['initialization'] = {'checkpoint_sha256': declared['initializer_checkpoint_sha256']}
        model = base._model_config(source_config)
        depth = int(declared['architecture'][-1])
        horizon,gain = {1:(4,40),2:(6,100),3:(8,360)}[depth]
        assert model['num_iterations_inference'] == model['num_iterations_training'] == declared['T'] == declared['K'] == horizon
        assert model['input_gain'] == gain and model['weight_min'] == 0 and model['weight_max'] == 100
        voltage, current = float(model['voltage_amp']), float(model['current_amp'])
        assert (voltage, current) == {'baseline': (1., 1.), 'legacy': (4., .25), 'ours': (4., 1.)}[declared['scheme']]
        case = dict(architecture=declared['architecture'], scheme=declared['scheme'], T=horizon, K=horizon, native_T=horizon, native_K=horizon,
                    base_beta=declared['injected_beta'] / (voltage / current)**depth,
                    injected_beta=declared['injected_beta'], voltage_amp=voltage, current_amp=current,
                    voltage_amplification=voltage, current_amplification=current, source_config=source_config,
                    reconstructed_initialization_tensor_sha256=declared['reconstructed_initialization_tensor_sha256'])
        selected = gate._selected(case, cfg['checkpoint_role'])
        run = out / ('smoke' if args.smoke else 'runs') / declared['name']
        if run.exists():
            raise FileExistsError(f'Refusing to overwrite an existing attempt: {run}')
        manifest = dict(study_id=cfg['study_id'], run_id=declared['name'], arm_id=declared['name'],
            evidence_class=cfg['evidence_class'], dataset='ordinary_mnist_validation', smoke=args.smoke,
            command=[sys.executable, *sys.argv], configuration=dict(path=str(args.config.resolve()),
                sha256=base.sha256_file(args.config), resolved=resolved),
            inputs=dict(declared, initializer_checkpoint=declared['initializer_checkpoint_path'],
                        initializer_sha256=declared['initializer_checkpoint_sha256']),
            git=dict(commit=cfg.get('source_commit') or subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                     runner_sha256=base.sha256_file(Path(__file__)), dirty_worktree=True),
            runtime=dict(base.runtime_context(target=args.target or cfg['target']), torch_version=torch.__version__,
                         cuda_version=torch.version.cuda, device=args.device),
            checkpoint_role=cfg['checkpoint_role'], checkpoint_epoch=0, model_seed=0,
            cohort_sha256=cohort['cohort_sha256'], official_test_read=False,
            optimizer_steps_applied=False, accuracy_evaluation=False)
        start_run(run, manifest)
        shutil.copy2(ROOT / declared['initializer_checkpoint_path'], run / 'checkpoints/initialization.pt')
        atomic_write_json(run / 'source_config.json', source_config)
        rows, residuals, signals, displacements, phases, guards, noise_records = [], [], [], [], [], [], []
        case_started = time.monotonic()
        print('START', declared['name'], 'batches', len(batches), flush=True)
        try:
            for batch in batches:
                check_deadline()
                cache.context = (case['architecture'],case['scheme'],int(batch['batch_index']),batch['payload_sha256'])
                output = shadow._run_precision(case=case, selected=selected, batch=batch,
                    precision='float64', device=torch.device(args.device), residual_threshold=.01,
                    amplification_exponent=depth, nudging_mode='current', eqprop_variant='centered',
                    capture_endpoint_states=True)
                guard = output['guard']
                assert guard['parameter_tensors_unchanged_after_cast']
                assert guard['frozen_current_force_matches_post_T_cost_gradient']
                assert guard['frozen_current_force_unchanged_across_phases']
                runtime = output['captured_runtime']
                assert all(bool((p.state == 0).all()) for p in runtime['parameters'] if str(p.name).startswith('Bias_'))
                before = base._parameter_state_sha256(runtime['parameters'])
                for name, gradient in output['bptt_gradients'].items():
                    key = (case['architecture'],case['scheme'], int(batch['batch_index']), name)
                    digest = base._tensor_sha256(gradient)
                    assert bptt_hashes.setdefault(key, digest) == digest, 'BPTT changed with beta'
                residual_passed = all(bool(r['gate_passed']) for r in output['residual_rows'])
                residuals.extend(output['residual_rows'])
                displacements.extend(output['displacement_rows'])
                phases.extend(output['phase_rows'])
                guards.append(guard)
                phase_states = output['captured_phase_states']
                for i, free in enumerate(output['captured_post_t_states']):
                    positive, negative = phase_states['positive'][i], phase_states['negative'][i]
                    rms = lambda v: float(v.square().mean().sqrt())
                    signals.append(dict(architecture=case['architecture'],scheme=case['scheme'], relative_target=declared['relative_target'], injected_beta=case['injected_beta'],
                        batch_index=int(batch['batch_index']), state_layer_index=i,
                        state_layer_name=str(runtime['free_layers'][i].name), element_count=free.numel(),
                        free_rms=rms(free), positive_free_rms=rms(positive-free),
                        negative_free_rms=rms(negative-free), centered_half_difference_rms=rms((positive-negative)/2)))
                assert list(output['gradients']) == declared['parameter_order']
                for sigma,seed,draw_index in draws:
                    noise = gate._read_noise_contract({})
                    noise.update(endpoint_read_noise_std=sigma,
                                 endpoint_read_noise_seed=seed)
                    scored, numerators, records, noise_guard = gate._apply_endpoint_read_noise(
                        output=output, case=case, batch=batch, noise_contract=noise,
                        device=torch.device(args.device))
                    assert noise_guard['parameter_tensors_unchanged'] and noise_guard['input_remained_exact']
                    for record in records:
                        check_noise_record(noise_hashes,record,architecture=case['architecture'],
                            sigma=sigma,seed=seed,batch_index=int(batch['batch_index']))
                        noise_records.append(dict(record,injected_beta=case['injected_beta'],
                            relative_target=declared['relative_target'],noise_seed=seed,noise_draw_index=draw_index))
                    output.update(scored_gradients=scored, scored_numerators=numerators, noise_contract=noise)
                    for name, clean in output['gradients'].items():
                        row = gate._layer_row(config=resolved, case=case, selected=selected, batch=batch,
                                              parameter_name=name, output=output, residual_passed=residual_passed)
                        bptt, noisy = output['bptt_gradients'][name], scored[name]
                        noise_norm = float((noisy-clean).norm())
                        clean_norm = float(clean.norm())
                        row.update(checkpoint_epoch=0, readout_sigma=sigma, noise_draw_index=draw_index,noise_seed=seed,
                            relative_target=declared['relative_target'],
                            calibrated_free_output_rms=declared['calibration']['free_rms'],
                            calibrated_centered_output_rms=declared['calibration']['centered_rms'],
                            calibrated_relative_output_rms=declared['calibration']['relative_output_rms'],
                            bptt_rms=float(bptt.square().mean().sqrt()),
                            clean_eqprop_rms=float(clean.square().mean().sqrt()),
                            eqprop_rms=float(noisy.square().mean().sqrt()),
                            added_noise_rms=noise_norm/math.sqrt(clean.numel()),
                            noise_over_clean_norm=noise_norm/clean_norm if clean_norm else None,
                            bptt_near_zero_fraction=float((bptt.abs() <= 1e-12).double().mean()),
                            clean_eqprop_near_zero_fraction=float((clean.abs() <= 1e-12).double().mean()),
                            eqprop_near_zero_fraction=float((noisy.abs() <= 1e-12).double().mean()),
                            bptt_gradient_sha256=bptt_hashes[(case['architecture'],case['scheme'], int(batch['batch_index']), name)])
                        rows.append(row)
                assert before == base._parameter_state_sha256(runtime['parameters'])
                append_metric(run / 'metrics.jsonl', dict(stage='initialization_replay', epoch=0,
                    dataset_split='validation', batch_index=int(batch['batch_index']), comparisons=len(rows)))
                update_status_progress(run, dict(stage='initialization_replay', batches_complete=int(batch['batch_index'])+1,
                                                 batches=len(batches), comparisons=len(rows)))
                del runtime,output
                gc.collect()
                if (int(batch['batch_index'])+1) % 6 == 0:
                    print('PROGRESS', declared['name'], int(batch['batch_index'])+1, '/', len(batches), flush=True)
            output_signals = [r for r in signals if r['state_layer_index']==depth]
            free_ss = sum(r['element_count']*r['free_rms']**2 for r in output_signals)
            centered_ss = sum(r['element_count']*r['centered_half_difference_rms']**2 for r in output_signals)
            actual_ratio = math.sqrt(centered_ss/free_ss)
            if not args.smoke:
                assert abs(actual_ratio/declared['relative_target']-1) <= .01, 'Physical output target mismatch'
                assert math.isclose(math.sqrt(free_ss/sum(r['element_count'] for r in output_signals)), declared['calibration']['free_rms'],rel_tol=1e-8)
            verify_inputs()
            assert len(rows) == len(batches)*len(draws)*(depth+1)
            assert len(noise_records) == len(batches)*sum(s > 0 for s,_,_ in draws)*2*(depth+1)
            assert base.sha256_file(run / 'checkpoints/initialization.pt') == declared['initializer_checkpoint_sha256']
            for name, records in [('layer_metrics', rows), ('equilibrium_residuals', residuals),
                                   ('state_signal', signals), ('state_displacement', displacements),
                                   ('phase_diagnostics', phases), ('endpoint_read_noise', noise_records)]:
                if records:
                    base._write_csv(run / f'{name}.csv', records)
            atomic_write_json(run / 'read_only_guards.json', guards)
            complete_run(run, terminal_metrics=dict(replay_batches=len(batches), layer_comparisons=len(rows),
                noise_tensor_draws=len(noise_records), source_bytes_unchanged=True,
                all_parameter_tensors_unchanged=True, all_bias_tensors_exact_zero=True,
                checkpoint_epoch=0, checkpoint_role=cfg['checkpoint_role'], official_test_read=False,
                actual_relative_output_rms=actual_ratio, relative_target=declared['relative_target'],
                cuda_peak_memory_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else 0,
                optimizer_steps_applied=False, accuracy_evaluation=False,
                undefined_cosines=sum(r['cosine'] is None for r in rows),
                all_endpoint_residual_gates_passed=all(r['gate_passed'] for r in residuals),
                elapsed_seconds=time.monotonic()-case_started),
                completion=dict(criteria_met=True, coverage_complete=True))
            assert not validate_run(run)
            summary.append(dict(name=declared['name'], run_dir=str(run.resolve()), state='complete',
                                layer_comparisons=len(rows), result_sha256=base.sha256_file(run / 'result.json')))
            print('DONE', declared['name'], round(time.monotonic()-case_started, 1), 'seconds', flush=True)
        except BaseException as exc:
            if rows:
                base._write_csv(run / 'partial_layer_metrics.csv', rows)
            fail_run(run, error=exc)
            raise
    expected_comparisons = sum(int(c['architecture'][-1])+1 for c in cfg['cases'])*len(batches)*len(draws)
    assert sum(c['layer_comparisons'] for c in summary) == expected_comparisons
    atomic_write_json(smoke_summary_path if args.smoke else out / 'summary.json', dict(
        state='complete', cases=summary, config_sha256=base.sha256_file(args.config),
        runner_sha256=base.sha256_file(Path(__file__)), cohort_sha256=cohort['cohort_sha256'],
        bptt_reference_computations=cache.computations+cache.parity_checks,bptt_cache_hits=cache.hits,
        bptt_cache_parity_checks=cache.parity_checks,
        bptt_invariant_across_beta=True, noise_draws_matched_across_schemes_and_beta=True,
        elapsed_seconds=time.monotonic()-started, total_including_smoke_seconds=time.monotonic()-started+prior_seconds,
        official_test_read=False, optimizer_steps_applied=False, accuracy_evaluation=False))
    print('COMPLETE', len(summary), 'cases', round(time.monotonic()-started, 1), 'seconds', flush=True)


if __name__ == '__main__':
    main()
