"""Replay saved Conv3 checkpoints across beta with matched endpoint read noise."""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def replay_contexts(cfg):
    """Run the largest K first when a common-reference K sweep is requested."""
    ks = cfg['K_values'] if 'K_values' in cfg else [cfg['K']]
    assert ks and ks == sorted(set(ks)) and all(isinstance(k, int) and k > 0 for k in ks)
    order = [ks[-1], *ks[:-1]] if 'K_values' in cfg else ks
    return [dict(case, replay_T=T, replay_K=K,
                 context_name=(case['name'] + f'_T{T}K{K}'
                 if 'T_values' in cfg or 'K_values' in cfg else case['name']))
            for case in cfg['cases'] for T in cfg.get('T_values', [cfg.get('T')]) for K in order]


def vector_comparison(candidate, reference):
    """Direction and magnitude against a fixed reference, preserving zero cases."""
    a, b = candidate.detach().cpu().reshape(-1), reference.detach().cpu().reshape(-1)
    an, bn = float(a.norm()), float(b.norm())
    return dict(cosine=float(a.dot(b)/(an*bn)) if an and bn else None,
                relative_l2=float((a-b).norm())/bn if bn else None,
                norm_ratio=an/bn if bn else None)


def beta_points(cfg, *, smoke=False, case=None):
    key = 'injected_betas' if 'injected_betas' in cfg else 'beta_factors'
    points = cfg[('smoke_' if smoke else '') + key]
    if case is not None and key in case:
        selected = case[key]
        assert selected and selected == sorted(set(selected))
        assert set(selected) <= set(cfg[key]), 'Case beta outside declared grid'
        points = [point for point in points if point in selected]
    return points


def cell_identity(declared, point, *, absolute=False):
    suffix = '_beta_' + format(point, '.12g') if absolute else '_factor_' + f'{point:g}'
    return declared['context_name'] + suffix.replace('.', 'p')


def check_noise_record(seen, record, *, sigma, seed, batch_index):
    """Check scaled noise bytes within sigma and the shared raw RNG identity.

    The frozen analyzer hashes (z*sigma)/sigma, which need not reproduce z
    bitwise. Across sigma, its deterministic seed and shape identify the same
    standard-normal draw; within sigma, require exact reconstructed bytes.
    """
    phase_index = {'negative': 1, 'positive': 2}[record['phase']]
    layer_index = record['state_layer_index']
    expected_seed = int(seed) + 1000 * batch_index + 100 * phase_index + layer_index
    assert record['seed'] == expected_seed, 'Unexpected endpoint RNG seed'
    raw_key = ('raw_rng', seed, batch_index, record['phase'], layer_index)
    identity = (expected_seed, record['element_count'])
    assert seen.setdefault(raw_key, identity) == identity, 'Unmatched raw RNG identity'
    scaled_key = ('scaled_bytes', sigma, *raw_key[1:])
    digest = record['standard_normal_sha256']
    assert seen.setdefault(scaled_key, digest) == digest, 'Unmatched endpoint noise within sigma'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--data-root')
    parser.add_argument('--target')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    assert 0 <= args.shard_index < args.shard_count
    out = ROOT / cfg['output_root']
    if cfg['study_id'] not in (ROOT / 'docs/current_simulations.md').read_text():
        raise RuntimeError('Register the planned result directory before running.')
    spec = importlib.util.spec_from_file_location(
        'trained_beta_noise_gate', ROOT / cfg['analyzer_source_root'] /
        'analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    base, shadow, torch = gate.base, gate.shadow, gate.torch
    from experiments.reporting import (
        append_metric, atomic_write_json, complete_run, fail_run,
        start_run, update_status_progress, validate_run,
    )

    def verify_inputs():
        for name, digest in cfg['analyzer_sha256'].items():
            assert base.sha256_file(ROOT / cfg['analyzer_source_root'] / name) == digest, name
        prior_path = ROOT / cfg['cohort_config']
        assert base.sha256_file(prior_path) == cfg['cohort_config_sha256']
        prior = json.loads(prior_path.read_text())
        for name, digest in prior['runtime_sha256'].items():
            assert base.sha256_file(ROOT / cfg['source_contract']['runtime_source_root'] / name) == digest, name
        for checkpoint in cfg['cases']:
            for name, digest in checkpoint['source_hashes'].items():
                assert base.sha256_file(ROOT / checkpoint['bundle'] / name) == digest, (checkpoint['name'], name)
        if 'initializer_checkpoint_path' in cfg:
            assert base.sha256_file(ROOT / cfg['initializer_checkpoint_path']) == cfg['initializer_checkpoint_sha256']
        return prior

    prior = verify_inputs()
    resolved = {**prior, **cfg, 'dataset': prior['dataset']}
    resolved['dataset'] = dict(resolved['dataset'], root=args.data_root or cfg.get('data_root', prior['dataset']['root']))
    assert len(cfg['cases']) == cfg['expected_checkpoints'] > 0
    points = beta_points(cfg)
    contexts = replay_contexts(cfg)
    absolute = 'injected_betas' in cfg
    assert points == sorted(set(points))
    assert all(math.isfinite(x) and x > 0 for x in points)
    assert all(c['replay_T'] > 0 for c in contexts)
    assert len(set(cfg['read_noise_seeds'])) == cfg['draws_per_noisy_batch'] == 4
    assert not cfg['official_test_read'] and not cfg['optimizer_steps_applied']
    started = time.monotonic()
    runner_hash = base.sha256_file(Path(__file__))
    config_hash = base.sha256_file(args.config)
    smoke_path = out / 'smoke_summary.json'
    prior_seconds = cfg.get('prior_attempt_seconds', 0.)
    if not args.smoke:
        smoke = json.loads(smoke_path.read_text())
        assert smoke['state'] == 'complete'
        assert smoke['config_sha256'] == config_hash and smoke['runner_sha256'] == runner_hash
        assert len(smoke['cells']) == sum(len(beta_points(cfg, smoke=True, case=c)) for c in contexts)
        for cell in smoke['cells']:
            assert not validate_run(ROOT / cell['run_dir'])
        prior_seconds = smoke['total_including_smoke_seconds']

    def check_deadline():
        if time.monotonic() - started + prior_seconds >= cfg['budget_seconds']:
            raise TimeoutError('Declared smoke plus replay budget exhausted.')

    dataset_source = json.loads((ROOT / cfg['cases'][0]['bundle'] / 'config.used.json').read_text())
    batches, cohort = gate._build_cohort(resolved, [{'source_config': dataset_source}])
    assert len(batches) == cfg['expected_batches'] == 36
    if args.smoke:
        batches = batches[:1]
    factors = beta_points(cfg, smoke=args.smoke)[args.shard_index::args.shard_count]
    assert factors, 'Empty beta shard'
    atomic_write_json(out / ('smoke_cohort.json' if args.smoke else 'cohort.json'),
                      dict(cohort, replayed_batch_indices=[int(b['batch_index']) for b in batches]))
    summary, bptt_hashes, noise_hashes = [], {}, {}
    reference_gradients, reference_context = {}, None
    for declared in contexts:
        identity = (declared['name'], declared['replay_T'])
        if identity != reference_context:
            reference_gradients.clear()
            reference_context = identity
        bundle = ROOT / declared['bundle']
        source_config = json.loads((bundle / 'config.used.json').read_text())
        assert source_config['training_algorithm'] == 'EP' and source_config['runtime_dtype'] == 'float64'
        assert math.isclose(source_config['eqprop']['injected_beta_B'], declared['beta'], rel_tol=1e-12)
        assert float(source_config['eqprop']['endpoint_read_noise_std']) == declared['sigma']
        gate._bias_lr_proof(source_config)
        model = base._model_config(source_config)
        assert model['num_iterations_inference'] == model['num_iterations_training'] == 8
        T = declared['replay_T']
        K = declared['replay_K']
        role = declared.get('checkpoint_role', 'final')
        epoch = declared.get('checkpoint_epoch', 30)
        assert role in ('final', 'best_validation', 'reconstructed_initialization')
        if role == 'reconstructed_initialization':
            assert epoch == 0
            checkpoint_sha = cfg['initializer_checkpoint_sha256']
            source_config['init_checkpoint_path'] = str(ROOT / cfg['initializer_checkpoint_path'])
            source_config['initialization'] = {'checkpoint_sha256': checkpoint_sha}
            checkpoint_fields = dict(reconstructed_initialization_tensor_sha256=
                                     declared['reconstructed_initialization_tensor_sha256'])
        else:
            checkpoint_file = 'final_model.pt' if role == 'final' else 'best_model.pt'
            weights_file = 'weights_final.npz' if role == 'final' else 'weights_best.npz'
            checkpoint_sha = declared['source_hashes'][checkpoint_file]
            checkpoint_fields = dict(native_checkpoint_dtype='float64',
                final_checkpoint_path=bundle / checkpoint_file, weights_final_path=bundle / weights_file,
                final_checkpoint_sha256=checkpoint_sha,
                best_checkpoint_path=bundle / checkpoint_file, weights_best_path=bundle / weights_file,
                best_checkpoint_sha256=checkpoint_sha)
        assert K > 0
        voltage, current = float(model['voltage_amp']), float(model['current_amp'])
        assert (voltage, current) == {'baseline': (1., 1.), 'ours': (4., 1.), 'legacy': (4., .25)}[declared['scheme']]
        for point in beta_points(cfg, smoke=args.smoke, case=declared)[args.shard_index::args.shard_count]:
            check_deadline()
            beta = point if absolute else declared['beta'] * point
            factor = beta / declared['beta']
            case = dict(architecture='conv3', scheme=declared['scheme'], T=T, K=K, native_T=8, native_K=8,
                base_beta=beta / (voltage / current)**3, injected_beta=beta,
                voltage_amp=voltage, current_amp=current, voltage_amplification=voltage,
                current_amplification=current, source_config=source_config, **checkpoint_fields)
            selected = gate._selected(case, role)
            cell_name = cell_identity(declared, point, absolute=absolute)
            run = out / ('smoke' if args.smoke else 'runs') / cell_name
            if run.exists():
                raise FileExistsError(f'Refusing to overwrite an existing attempt: {run}')
            manifest = dict(study_id=cfg['study_id'], run_id=cell_name, arm_id=cell_name,
                evidence_class='ordinary_mnist_selection', evidence_tier='exploratory',
                dataset='ordinary_mnist_validation', smoke=args.smoke,
                command=[sys.executable, *sys.argv], configuration=dict(path=str(args.config),
                    sha256=config_hash, resolved=cfg), inputs=declared,
                git=dict(commit=cfg['source_commit'], runner_sha256=runner_hash, dirty_worktree=True),
                runtime=dict(base.runtime_context(target=args.target or cfg['target']), torch_version=torch.__version__,
                             cuda_version=torch.version.cuda, device=args.device),
                checkpoint_role=role, checkpoint_epoch=epoch, beta_factor=factor,
                injected_beta=beta, base_beta=case['base_beta'],
                cohort_sha256=cohort['cohort_sha256'], official_test_read=False,
                optimizer_steps_applied=False, accuracy_evaluation=False,
                shard_index=args.shard_index, shard_count=args.shard_count,
                replay_T=T, replay_K=K,
                comparison_reference_K=max(cfg['K_values']) if 'K_values' in cfg else None)
            start_run(run, manifest)
            atomic_write_json(run / 'source_config.json', source_config)
            rows, signals, residuals, guards = [], [], [], []
            cell_started = time.monotonic()
            print('START', cell_name, 'beta', beta, 'batches', len(batches), flush=True)
            try:
                for batch in batches:
                    check_deadline()
                    output = shadow._run_precision(case=case, selected=selected, batch=batch,
                        precision='float64', device=torch.device(args.device), residual_threshold=.01,
                        amplification_exponent=3, nudging_mode='current', eqprop_variant='centered',
                        capture_endpoint_states=True)
                    guard = output['guard']
                    assert guard['parameter_tensors_unchanged_after_cast']
                    assert guard['frozen_current_force_matches_post_T_cost_gradient']
                    assert guard['frozen_current_force_unchanged_across_phases']
                    runtime = output['captured_runtime']
                    assert all(bool((p.state == 0).all()) for p in runtime['parameters'] if str(p.name).startswith('Bias_'))
                    before = base._parameter_state_sha256(runtime['parameters'])
                    for name, gradient in output['bptt_gradients'].items():
                        key = (declared['context_name'], int(batch['batch_index']), name)
                        digest = base._tensor_sha256(gradient)
                        assert bptt_hashes.setdefault(key, digest) == digest, 'BPTT changed with beta'
                    if 'K_values' in cfg and K == max(cfg['K_values']):
                        for name in output['gradients']:
                            reference_gradients[(point, int(batch['batch_index']), name)] = (
                                output['bptt_gradients'][name].detach().cpu().clone(),
                                output['gradients'][name].detach().cpu().clone())
                    residual_pass = all(bool(r['gate_passed']) for r in output['residual_rows'])
                    residuals.extend(output['residual_rows'])
                    guards.append(guard)
                    for i, free in enumerate(output['captured_post_t_states']):
                        positive = output['captured_phase_states']['positive'][i]
                        negative = output['captured_phase_states']['negative'][i]
                        zero = output['captured_zero_states'][i]
                        def rms(value):
                            return float(value.square().mean().sqrt())
                        is_hidden = i < len(output['captured_post_t_states']) - 1
                        active = lambda v: v.abs() > 1e-8
                        signals.append(dict(checkpoint=declared['name'], scheme=declared['scheme'], T=T, K=K,
                            training_sigma=declared['sigma'] if epoch else None, beta_factor=factor, injected_beta=beta,
                            batch_index=int(batch['batch_index']), layer_index=i,
                            layer_name=str(runtime['free_layers'][i].name), element_count=free.numel(),
                            free_rms=rms(free), positive_free_rms=rms(positive-free),
                            negative_free_rms=rms(negative-free), centered_half_difference_rms=rms((positive-negative)/2),
                            zero_free_rms=rms(zero-free), midpoint_minus_zero_rms=rms((positive+negative)/2-zero),
                            positive_activity_change_fraction=float((active(positive) != active(free)).double().mean()) if is_hidden else None,
                            negative_activity_change_fraction=float((active(negative) != active(free)).double().mean()) if is_hidden else None,
                            phase_activity_disagreement_fraction=float((active(positive) != active(negative)).double().mean()) if is_hidden else None))
                    draws = [(0., None)] + [(declared['sigma'], seed) for seed in cfg['read_noise_seeds']]
                    for draw_index, (sigma, seed) in enumerate(draws):
                        noise = gate._read_noise_contract({})
                        noise.update(endpoint_read_noise_std=sigma, endpoint_read_noise_seed=seed)
                        scored, numerators, records, noise_guard = gate._apply_endpoint_read_noise(
                            output=output, case=case, batch=batch, noise_contract=noise, device=torch.device(args.device))
                        assert noise_guard['parameter_tensors_unchanged'] and noise_guard['input_remained_exact']
                        for record in records:
                            check_noise_record(noise_hashes, record, sigma=sigma, seed=seed,
                                               batch_index=int(batch['batch_index']))
                        output.update(scored_gradients=scored, scored_numerators=numerators, noise_contract=noise)
                        for name, clean in output['gradients'].items():
                            row = gate._layer_row(config=resolved, case=case, selected=selected, batch=batch,
                                parameter_name=name, output=output, residual_passed=residual_pass)
                            bptt, noisy = output['bptt_gradients'][name], scored[name]
                            noise_norm, clean_norm = float((noisy-clean).norm()), float(clean.norm())
                            row.update(checkpoint=declared['name'], training_sigma=declared['sigma'] if epoch else None,
                                training_gpu=declared['gpu'] if epoch else None,
                                training_beta=declared['beta'] if epoch else None, beta_factor=factor,
                                checkpoint_epoch=epoch, checkpoint_sha256=checkpoint_sha,
                                source_bundle=declared['bundle'], noise_draw_index=draw_index,
                                bptt_rms=float(bptt.square().mean().sqrt()), clean_eqprop_rms=float(clean.square().mean().sqrt()),
                                eqprop_rms=float(noisy.square().mean().sqrt()), added_noise_rms=noise_norm/math.sqrt(clean.numel()),
                                noise_over_clean_norm=noise_norm/clean_norm if clean_norm else None,
                                bptt_near_zero_fraction=float((bptt.abs() <= 1e-12).double().mean()),
                                clean_eqprop_near_zero_fraction=float((clean.abs() <= 1e-12).double().mean()),
                                eqprop_near_zero_fraction=float((noisy.abs() <= 1e-12).double().mean()),
                                bptt_gradient_sha256=bptt_hashes[(declared['context_name'], int(batch['batch_index']), name)])
                            if 'K_values' in cfg:
                                ref_bptt, ref_ep = reference_gradients[(point, int(batch['batch_index']), name)]
                                row['reference_K'] = max(cfg['K_values'])
                                row['reference_bptt_sha256'] = base._tensor_sha256(ref_bptt)
                                for prefix, value, reference in [
                                        ('bptt_vs_reference', bptt, ref_bptt),
                                        ('clean_ep_vs_reference_ep', clean, ref_ep),
                                        ('ep_vs_reference_bptt', noisy, ref_bptt)]:
                                    row.update({prefix+'_'+metric: value for metric, value in
                                                vector_comparison(value, reference).items()})
                            rows.append(row)
                    assert before == base._parameter_state_sha256(runtime['parameters'])
                    append_metric(run / 'metrics.jsonl', dict(stage='trained_beta_replay', epoch=epoch,
                        dataset_split='validation', batch_index=int(batch['batch_index']), comparisons=len(rows)))
                    update_status_progress(run, dict(stage='trained_beta_replay', batches_complete=int(batch['batch_index'])+1,
                                                     batches=len(batches), beta_factor=factor))
                    del output, runtime
                    gc.collect()
                for name, digest in declared['source_hashes'].items():
                    assert base.sha256_file(bundle / name) == digest, name
                if role == 'reconstructed_initialization':
                    assert base.sha256_file(ROOT / cfg['initializer_checkpoint_path']) == checkpoint_sha
                assert len(rows) == len(batches) * 5 * 4
                for name, records in [('layer_metrics', rows), ('state_signals', signals), ('equilibrium_residuals', residuals)]:
                    base._write_csv(run / f'{name}.csv', records)
                atomic_write_json(run / 'read_only_guards.json', guards)
                elapsed = time.monotonic()-cell_started
                complete_run(run, terminal_metrics=dict(replay_batches=len(batches), layer_comparisons=len(rows),
                    source_bytes_unchanged=True, all_parameter_tensors_unchanged=True,
                    all_bias_tensors_exact_zero=True, bptt_invariant_across_beta=True,
                    noise_draws_matched=True, checkpoint_epoch=epoch, checkpoint_role=role,
                    official_test_read=False, optimizer_steps_applied=False, accuracy_evaluation=False,
                    undefined_cosines=sum(r['cosine'] is None for r in rows),
                    residual_failed_rows=sum(not r['gate_passed'] for r in residuals),
                    elapsed_seconds=elapsed), completion=dict(criteria_met=True, coverage_complete=True))
                assert not validate_run(run)
                summary.append(dict(name=cell_name, run_dir=str(run.relative_to(ROOT)), state='complete',
                    layer_comparisons=len(rows), result_sha256=base.sha256_file(run / 'result.json'), elapsed_seconds=elapsed))
                atomic_write_json(out / ('smoke_progress.json' if args.smoke else 'progress.json'),
                                  dict(cells=summary, cells_expected=sum(len(beta_points(cfg, smoke=args.smoke, case=c)[args.shard_index::args.shard_count]) for c in contexts)))
                print('DONE', cell_name, round(elapsed, 2), 'seconds', flush=True)
            except BaseException as exc:
                if rows:
                    base._write_csv(run / 'partial_layer_metrics.csv', rows)
                fail_run(run, error=exc)
                raise
    verify_inputs()
    expected = sum(len(beta_points(cfg, smoke=args.smoke, case=c)[args.shard_index::args.shard_count]) for c in contexts) * len(batches) * 5 * 4
    assert sum(c['layer_comparisons'] for c in summary) == expected
    atomic_write_json(smoke_path if args.smoke else out / 'summary.json', dict(state='complete', cells=summary,
        shard_index=args.shard_index, shard_count=args.shard_count, beta_points=factors,
        beta_mode='injected' if absolute else 'factor',
        replay_T=cfg.get('T_values', cfg.get('T')), replay_K=cfg.get('K_values', cfg.get('K')), target=args.target or cfg['target'],
        config_sha256=config_hash, runner_sha256=runner_hash, cohort_sha256=cohort['cohort_sha256'],
        bptt_invariant_across_beta=True, noise_draws_matched_across_schemes_and_beta=True,
        elapsed_seconds=time.monotonic()-started, total_including_smoke_seconds=time.monotonic()-started+prior_seconds,
        official_test_read=False, optimizer_steps_applied=False, accuracy_evaluation=False))
    print('COMPLETE', len(summary), 'cells', round(time.monotonic()-started, 2), 'seconds', flush=True)


if __name__ == '__main__':
    main()
