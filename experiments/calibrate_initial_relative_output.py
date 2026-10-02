"""Match initial centered physical-output displacement, then emit exact training configs.

Uses the archived frozen-current runtime and saved seed-0 initializers. Calibration
is noiseless; the emitted training configs use the declared endpoint read noise.
No optimizer, gradient-fidelity search, or T/K variation occurs here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pooled_ratio(rows):
    """Ratio of pooled RMS values, never the mean of per-batch ratios."""
    free = sum(row['free_sum_squares'] for row in rows)
    displacement = sum(row['centered_sum_squares'] for row in rows)
    count = sum(row['elements'] for row in rows)
    if count <= 0 or free <= 0 or displacement <= 0:
        raise ValueError('Relative output displacement requires positive finite pooled norms.')
    result = dict(free_rms=math.sqrt(free/count), centered_rms=math.sqrt(displacement/count),
                  relative_output_rms=math.sqrt(displacement/free), elements=count)
    if not all(math.isfinite(x) for x in result.values()):
        raise FloatingPointError('Nonfinite pooled output measurement.')
    return result


def matched_beta(measure, beta, target, tolerance, max_corrections):
    """One measured initial guess followed by at most the declared corrections."""
    history = []
    for correction in range(max_corrections + 1):
        measured = measure(beta)
        error = abs(measured['relative_output_rms']/target - 1)
        history.append(dict(injected_beta=beta, correction=correction,
                            relative_target_error=error, **measured))
        if error <= tolerance:
            return beta, history
        beta *= target/measured['relative_output_rms']
        if not math.isfinite(beta) or beta <= 0:
            raise FloatingPointError('Invalid corrected beta.')
    raise RuntimeError(f'Target {target} missed after {max_corrections} corrections: {history[-1]}')


def training_config(source, declared, cfg, *, beta, target, calibration):
    c = copy.deepcopy(source)
    architecture, scheme = declared['architecture'], declared['scheme']
    name = f'{architecture}_{scheme}_relative{target:g}_sigma5em4_seed0'.replace('.', 'p')
    scale = (c['model_base']['voltage_amp']/c['model_base']['current_amp'])**int(architecture[-1])
    c.update(study_id=cfg['study_id'], arm_id=name, training_algorithm='EP',
             runtime_dtype='float64', beta=beta/scale,
             init_checkpoint_path=str(ROOT/declared['initializer_checkpoint_path']))
    c['initialization'] = dict(checkpoint_sha256=declared['initializer_checkpoint_sha256'],
                               recipe='wide_kaiming', seed=0, optimizer_state='fresh')
    c['lab']['epochs'] = 10
    c['model_base'].update(num_iterations_inference=declared['T'], num_iterations_training=declared['K'])
    c['eqprop'] = dict(variant='centered', nudging_mode='current', current_scale='auto',
                      normalize_current_scale=True, amplification_factor=scale, injected_beta_B=beta,
                      endpoint_read_noise_std=0.0005, endpoint_read_noise_seed=2026081601,
                      input_read_noise=False, phase_noise_correlation='independent',
                      layer_noise_correlation='independent', matched_standard_normal_draws_across_cases=True,
                      noise_applied_after_equilibrium_before_local_energy_gradient=True,
                      beta_policy='initial_centered_physical_output_relative_rms_fixed_during_training')
    c['evaluation'] = dict(checkpoint_selection='maximum_validation_accuracy', epoch_split='validation',
                          official_test={'policy':'disabled'}, inclusion_rule='Retain all scientific outcomes; compare final validation at epoch 10.')
    c['reporting'] = dict(study_id=cfg['study_id'], arm_id=name, evidence_class='ordinary_mnist_selection', paper_facing=False)
    c['relative_output_calibration'] = dict(target=target, definition=cfg['ratio_definition'],
        injected_beta=beta, base_beta=beta/scale, measured=calibration,
        source_config=declared['source_config'], source_config_sha256=declared['source_config_sha256'],
        source_T=source['model_base']['num_iterations_inference'], T=declared['T'], K=declared['K'])
    c['datasets']['mnist']['params']['root'] = cfg['data_root']
    c['max_batches'] = c['max_test_batches'] = None
    assert c['lr'] == source['lr'] == c['optimizer']['learning_rate']
    assert all(v == 0 for k,v in c['learning_rates_by_parameter'].items() if k.startswith('Bias_'))
    return c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--architecture', choices=('conv1','conv2','conv3'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--target', required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if not (ROOT/cfg['assigned_experiment']).is_file():
        raise RuntimeError('Missing assigned experiment record.')
    spec = importlib.util.spec_from_file_location('relative_output_gate', ROOT/cfg['analyzer_source_root']/'analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    base, shadow, torch = gate.base, gate.shadow, gate.torch
    from experiments.reporting import append_metric, atomic_write_json, complete_run, fail_run, start_run, update_status_progress, validate_run
    from experiments.exact_run import load_exact_config
    def verify():
        for path,digest in cfg['source_sha256'].items():
            if sha(ROOT/path) != digest:
                raise ValueError(f'Source bytes changed: {path}')
        for declared in cfg['cases']:
            for key in ('source_config','initializer_checkpoint'):
                path = declared[key] if key == 'source_config' else declared[key+'_path']
                if sha(ROOT/path) != declared[key+'_sha256']:
                    raise ValueError(f'Input bytes changed: {path}')
    verify()
    selected = [c for c in cfg['cases'] if args.architecture in (None,c['architecture'])]
    suffix = args.architecture or 'all'
    out = ROOT/cfg['output_root']/'calibration'/('smoke_'+suffix if args.smoke else suffix)
    if out.exists():
        raise FileExistsError(out)
    start = time.monotonic()
    start_run(out, dict(study_id=cfg['study_id'],run_id=out.name,arm_id=out.name,
        evidence_class='ordinary_mnist_selection',dataset='ordinary_mnist_validation',smoke=args.smoke,
        command=[sys.executable,*sys.argv],configuration=dict(path=str(args.config),sha256=sha(args.config),resolved=cfg),
        git=dict(commit=cfg['source_commit'],runner_sha256=sha(__file__),dirty_worktree=True),
        runtime=dict(base.runtime_context(target=args.target),device=args.device,torch_version=torch.__version__),
        official_test_read=False,optimizer_steps_applied=False))
    measurements, matches = [], []
    try:
        prior = json.loads((ROOT/cfg['cohort_config']).read_text())
        prior['dataset']['root'] = cfg['data_root']
        source = json.loads((ROOT/selected[0]['source_config']).read_text())
        batches,cohort = gate._build_cohort(prior,[{'source_config':source}])
        assert len(batches) == 36
        if args.smoke:
            batches = batches[:1]
        atomic_write_json(out/'cohort.json',cohort)
        for declared in selected:
            source = json.loads((ROOT/declared['source_config']).read_text())
            gate._bias_lr_proof(source)
            runtime_config = copy.deepcopy(source)
            runtime_config['init_checkpoint_path'] = str(ROOT/declared['initializer_checkpoint_path'])
            runtime_config['initialization'] = {'checkpoint_sha256':declared['initializer_checkpoint_sha256']}
            runtime_config['model_base'].update(num_iterations_inference=declared['T'],num_iterations_training=declared['K'])
            runtime = base._build_runtime(runtime_config,device=torch.device(args.device),gradient_iterations=declared['K'],nudging_mode='current',current_scale='auto')
            historical_initialization_hash = base._historical_initialization_sha256(runtime['parameters'])
            shadow._convert_runtime_dtype(runtime,torch.float64)
            parameters = runtime['parameters']
            initial_hash = base._parameter_state_sha256(parameters)
            assert all(p.state.dtype == torch.float64 for p in parameters)
            assert all(bool((p.state == 0).all()) for p in parameters if str(p.name).startswith('Bias_'))
            scale = (source['model_base']['voltage_amp']/source['model_base']['current_amp'])**int(declared['architecture'][-1])
            assert math.isclose(runtime['nudging_current_scale'],scale,rel_tol=1e-12)
            output_layer = runtime['cost_fn'].layers()[-1]
            assert tuple(output_layer.shape) == (20,)
            cache = {}
            def measure(beta):
                if beta in cache:
                    return cache[beta]
                rows = []
                for batch in batches:
                    if time.monotonic()-start > cfg['calibration_budget_seconds']:
                        raise TimeoutError('Calibration budget exhausted.')
                    shadow._set_input_and_reset(runtime,batch['images'].to(device=args.device,dtype=torch.float64),dtype=torch.float64)
                    runtime['minimizer_inference'].compute_equilibrium()
                    runtime['cost_fn'].set_target(batch['labels'].to(args.device))
                    states = base._clone_states(runtime['free_layers'])
                    expected_force = -runtime['cost_fn']._grad(output_layer,mean=False).detach().clone()
                    runtime['augmented_fn'].prepare_nudging()
                    force = runtime['augmented_fn']._nudging._force
                    assert torch.equal(force,expected_force)
                    endpoints = []
                    for sign in (-1,1):
                        base._restore_states(runtime['free_layers'],states)
                        runtime['augmented_fn'].nudging = torch.as_tensor(sign*beta/scale,dtype=torch.float64,device=args.device)
                        runtime['minimizer_augmented'].compute_equilibrium()
                        assert all(bool(torch.isfinite(x.state).all()) for x in runtime['free_layers'])
                        endpoints.append(runtime['free_layers'][-1].state.detach().clone())
                        assert torch.equal(runtime['augmented_fn']._nudging._force,expected_force)
                    free = states[-1]
                    centered = (endpoints[1]-endpoints[0])/2
                    rows.append(dict(batch_index=int(batch['batch_index']),elements=free.numel(),
                        free_sum_squares=float(free.square().sum()),centered_sum_squares=float(centered.square().sum())))
                assert base._parameter_state_sha256(parameters) == initial_hash
                value = pooled_ratio(rows)
                cache[beta] = value
                record = dict(architecture=declared['architecture'],scheme=declared['scheme'],injected_beta=beta,base_beta=beta/scale,
                              T=declared['T'],K=declared['K'],batches=rows,**value)
                measurements.append(record)
                atomic_write_json(out/'measurements.json',measurements)
                append_metric(out/'metrics.jsonl',dict(stage='initial_relative_output',dataset_split='validation',**{k:v for k,v in record.items() if k!='batches'}))
                update_status_progress(out,dict(stage='initial_relative_output',measurements=len(measurements),architecture=declared['architecture'],scheme=declared['scheme']))
                print('MEASURE',declared['architecture'],declared['scheme'],beta,value['relative_output_rms'],flush=True)
                return value
            pilot_beta = declared['pilot_injected_beta']
            pilot = measure(pilot_beta)
            if args.smoke:
                # Verify the lean endpoint measurement against the established
                # absolute-output replay on the identical fixed T/K and batch.
                voltage = source['model_base']['voltage_amp']
                current = source['model_base']['current_amp']
                case = dict(architecture=declared['architecture'],scheme=declared['scheme'],
                    T=declared['T'],K=declared['K'],native_T=declared['T'],native_K=declared['K'],
                    base_beta=pilot_beta/scale,injected_beta=pilot_beta,
                    voltage_amp=voltage,current_amp=current,
                    voltage_amplification=voltage,current_amplification=current,
                    source_config=runtime_config,
                    reconstructed_initialization_tensor_sha256=historical_initialization_hash)
                replay = shadow._run_precision(case=case,
                    selected=gate._selected(case,'reconstructed_initialization'),batch=batches[0],
                    precision='float64',device=torch.device(args.device),residual_threshold=.01,
                    amplification_exponent=int(declared['architecture'][-1]),
                    nudging_mode='current',eqprop_variant='centered',capture_endpoint_states=True)
                free = replay['captured_post_t_states'][-1]
                centered = (replay['captured_phase_states']['positive'][-1]-replay['captured_phase_states']['negative'][-1])/2
                reference = pooled_ratio([dict(free_sum_squares=float(free.square().sum()),
                    centered_sum_squares=float(centered.square().sum()),elements=free.numel())])
                assert all(math.isclose(pilot[key],reference[key],rel_tol=1e-12,abs_tol=1e-14)
                           for key in ('free_rms','centered_rms','relative_output_rms'))
                atomic_write_json(out/f"{declared['architecture']}_{declared['scheme']}_replay_parity.json",
                    dict(matched=True,lean=pilot,established_replay=reference,T=declared['T'],K=declared['K']))
                del replay
            for target in (cfg['relative_targets'][:1] if args.smoke else cfg['relative_targets']):
                beta,history = matched_beta(measure,pilot_beta*target/pilot['relative_output_rms'],target,cfg['relative_tolerance'],cfg['max_corrections'])
                match = dict(architecture=declared['architecture'],scheme=declared['scheme'],target=target,injected_beta=beta,base_beta=beta/scale,history=history)
                matches.append(match)
                if not args.smoke:
                    c = training_config(source,declared,cfg,beta=beta,target=target,calibration=history[-1])
                    path = ROOT/cfg['training_config_root']/(c['arm_id']+'.json')
                    if path.exists():
                        raise FileExistsError(path)
                    atomic_write_json(path,c)
                    load_exact_config(path)
                    match.update(training_config=str(path.relative_to(ROOT)),training_config_sha256=sha(path))
                atomic_write_json(out/'matches.json',matches)
        verify()
        assert len(matches) == len(selected)*(1 if args.smoke else len(cfg['relative_targets']))
        complete_run(out,terminal_metrics=dict(matches=len(matches),measurements=len(measurements),
            maximum_relative_target_error=max(m['history'][-1]['relative_target_error'] for m in matches),
            cohort_sha256=cohort['cohort_sha256'],official_test_read=False,optimizer_steps_applied=False,
            all_parameters_unchanged=True,all_bias_tensors_exact_zero=True,elapsed_seconds=time.monotonic()-start),
            completion=dict(criteria_met=True,coverage_complete=True))
        assert not validate_run(out)
    except BaseException as error:
        fail_run(out,error=error)
        raise


if __name__ == '__main__':
    main()
