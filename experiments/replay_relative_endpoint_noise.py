"""Conv3 initialization replay with absolute or nodewise-relative endpoint noise."""
from __future__ import annotations

import argparse
import copy
import gc
import csv
import os
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def configure_cuda(torch,cfg):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback is forbidden.')
    cap=cfg['cuda_allocator_cap_bytes']
    assert cap == int(7.5*1024**3)
    total=torch.cuda.get_device_properties(0).total_memory
    assert 0 < cap < total, 'CUDA allocator cap exceeds device capacity'
    torch.cuda.set_per_process_memory_fraction(cap/total,device=0)
    return cap,total


def assert_cuda_runtime(runtime):
    """Saved autograd buffers may move to CPU; live network tensors must not."""
    layers=list(runtime['energy_fn'].layers())+list(runtime['free_layers'])
    assert layers and all(layer.state.device.type=='cuda' for layer in layers), 'Network layer fell back to CPU'
    assert runtime['parameters'] and all(p.state.device.type=='cuda' for p in runtime['parameters']), 'Parameter fell back to CPU'


def offloaded_call(torch, function, *args, **kwargs):
    """Offload saved tensors for the entire operation, including residual autograd."""
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        return function(*args, **kwargs)


def noise_draws(cfg):
    assert cfg['read_noise_seeds'] == [101000000,102000000,103000000]
    assert cfg['noise_levels'] == [1e-4,1e-3]
    assert cfg['noise_models'] == ['absolute','nodewise_relative']
    return [('clean',0.,None,-1)] + [(model,level,seed,i)
        for model in cfg['noise_models'] for level in cfg['noise_levels']
        for i,seed in enumerate(cfg['read_noise_seeds'])]


def noisy_endpoint(torch, exact, xi, *, model, level):
    """Apply the declared nodewise formula; no floor and no input involvement."""
    assert exact.dtype == xi.dtype == torch.float64 and exact.shape == xi.shape
    assert math.isfinite(level) and level > 0
    if model == 'absolute':
        noise = xi*level
        expected_rms = level
    elif model == 'nodewise_relative':
        noise = (level*exact.abs())*xi
        expected_rms = level*float(exact.square().mean().sqrt())
        assert bool((noise[exact == 0] == 0).all())
    else:
        raise ValueError(f'Unknown acquisition model: {model}')
    acquired = exact + noise
    assert bool(torch.isfinite(acquired).all()) and bool(torch.isfinite(noise).all())
    return acquired,noise,expected_rms


def check_noise_record(seen, record, *, architecture, seed, batch_index):
    phase_index = {'negative':1,'positive':2}[record['phase']]
    layer = record['state_layer_index']
    expected = seed + 1000*batch_index + 100*phase_index + layer
    assert record['seed'] == expected
    key=(architecture,seed,batch_index,record['phase'],layer)
    identity=(record['element_count'],record['standard_normal_sha256'])
    assert seen.setdefault(key,identity)==identity, 'Raw Gaussian draws differ across models/levels/schemes'


def apply_read_noise(gate, *, output, case, batch, model, level, seed, device):
    torch,base,shadow = gate.torch,gate.base,gate.shadow
    if model == 'clean':
        assert level == 0 and seed is None
        return dict(output['gradients']),dict(output['numerators']),[],dict(parameter_tensors_unchanged=True,input_remained_exact=True,clean_endpoints_unchanged=True)
    runtime=output['captured_runtime'];assert_cuda_runtime(runtime);states=output['captured_phase_states']
    input_state=output['captured_input_state']
    before=base._parameter_state_sha256(runtime['parameters'])
    input_hash=base._tensor_sha256(input_state)
    endpoint_hashes={phase:[base._tensor_sha256(v) for v in values] for phase,values in states.items()}
    force_hash=base._tensor_sha256(runtime['augmented_fn']._nudging._force)
    endpoints={};records=[]
    for phase,phase_index in (('negative',1),('positive',2)):
        noised=[]
        for layer,clean in enumerate(states[phase]):
            exact=clean.detach().cpu().to(torch.float64)
            actual_seed=seed+1000*int(batch['batch_index'])+100*phase_index+layer
            generator=torch.Generator(device='cpu');generator.manual_seed(actual_seed)
            xi=torch.randn(tuple(exact.shape),generator=generator,dtype=torch.float64)
            acquired,noise,expected=noisy_endpoint(torch,exact,xi,model=model,level=level)
            noised.append(acquired)
            positive,negative=states['positive'][layer],states['negative'][layer]
            P=float(((positive.square().mean()+negative.square().mean())/2).sqrt())
            D=float(((positive-negative)/2).square().mean().sqrt())
            records.append(dict(architecture=case['architecture'],scheme=case['scheme'],
                batch_index=int(batch['batch_index']),phase=phase,state_layer_index=layer,
                state_layer_name=str(runtime['free_layers'][layer].name),element_count=exact.numel(),
                seed=actual_seed,noise_model=model,noise_level=level,
                level_units='voltage' if model=='absolute' else 'dimensionless',
                standard_normal_sha256=base._tensor_sha256(xi),
                clean_endpoint_rms=float(exact.square().mean().sqrt()),
                expected_noise_rms=expected,actual_noise_rms=float(noise.square().mean().sqrt()),
                empirical_noise_rms=float(noise.square().mean().sqrt()),
                empirical_noise_mean=float(noise.mean()),empirical_noise_maximum_absolute=float(noise.abs().max()),
                expected_noise_sum_squares=exact.numel()*expected**2,
                actual_noise_sum_squares=float(noise.square().sum()),
                clean_endpoint_sum_squares=float(exact.square().sum()),
                endpoint_rms=P,endpoint_rms_P=P,centered_half_difference_rms_D=D,D_over_P=D/P if P else None,
                clean_zero_count=int((exact==0).sum()),
                noise_nonzero_at_clean_zero_count=int((noise[exact==0]!=0).sum())))
        endpoints[phase]=gate._weight_dict(runtime,gate._energy_gradients_at_states(
            runtime,input_state=input_state,states=noised,device=device))
    scored={};numerators={}
    for name in output['gradients']:
        numerator,estimate=shadow._native_eqprop_estimate(eqprop_variant='centered',
            beta=float(case['base_beta']),denominator_scale=float(output['guard']['output_row_current_scale']),
            zero=output['endpoint_gradients']['zero'][name],negative=endpoints['negative'][name],positive=endpoints['positive'][name])
        assert estimate.dtype == torch.float64 and bool(torch.isfinite(estimate).all())
        scored[name]=estimate.detach().cpu().clone();numerators[name]=numerator.detach().cpu().clone()
    guard=dict(parameter_tensors_unchanged=before==base._parameter_state_sha256(runtime['parameters']),
        input_remained_exact=input_hash==base._tensor_sha256(runtime['energy_fn'].layers()[0].state),
        clean_endpoints_unchanged=endpoint_hashes=={p:[base._tensor_sha256(v) for v in values] for p,values in states.items()},
        frozen_force_unchanged=force_hash==base._tensor_sha256(runtime['augmented_fn']._nudging._force))
    assert_cuda_runtime(runtime)
    assert all(guard.values()),guard
    return scored,numerators,records,guard


def load_diagnostic_csv(path):
    """Load pinned diagnostic rows without converting string identities."""
    with path.open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        for key,value in row.items():
            if value in ('True','False'):
                row[key]=value=='True'
            elif value=='':
                row[key]=None
            elif not key.endswith('sha256'):
                try: row[key]=json.loads(value)
                except (ValueError,TypeError): pass
    return rows


def load_residual_reference(path):
    rows=load_diagnostic_csv(path)
    expected={(batch,phase,layer) for batch in range(36)
              for phase in ('post_T_free','zero','negative','positive') for layer in range(4)}
    actual={(r['batch_index'],r['phase'],r['layer_index']) for r in rows}
    assert len(rows)==len(expected) and actual==expected, 'Incomplete prior residual coverage'
    assert all(r['coverage_complete'] is True and r['example_count']==r['expected_example_count']==r['unique_source_index_count']==16 for r in rows)
    return rows


def load_phase_reference(path):
    rows=load_diagnostic_csv(path)
    expected={(batch,phase) for batch in range(36) for phase in ('zero','negative','positive')}
    assert len(rows)==len(expected) and {(r['batch_index'],r['phase']) for r in rows}==expected
    assert all(r['all_states_finite'] is True for r in rows)
    return rows


def reused_phase_residual(runtime,base,rows,screen_guard):
    state_hash=base._layer_state_sha256(runtime['free_layers'])
    beta=float(runtime['augmented_fn'].nudging)
    matching=[r for r in rows if r['state_sha256']==state_hash and r['actual_beta']==beta]
    assert len(matching)==1, 'Phase residual reuse endpoint/beta differs'
    assert base._parameter_state_sha256(runtime['parameters'])==screen_guard['parameter_state_sha256_after_cast']
    force=runtime['augmented_fn']._nudging._force
    assert base._tensor_sha256(force)==screen_guard['frozen_current_force_sha256']
    row=matching[0]
    return {key:row[key] for key in ('projected_kkt_rms','projected_kkt_max_abs','active_fraction')}


class ReusedResidualAccumulator:
    """Supply pinned diagnostics only for independently verified identical endpoints."""
    def __init__(self,rows,base,screen_guard):
        self.rows=rows;self.base=base;self.screen_guard=screen_guard;self.seen=set()
    def add_endpoint(self,runtime,*,gradient_source,phase,beta,source_indices):
        key={'post_T_free':'common_post_T_state_sha256','zero':'zero_state_sha256',
             'negative':'negative_state_sha256','positive':'positive_state_sha256'}[phase]
        assert self.base._layer_state_sha256(runtime['free_layers'])==self.screen_guard[key], 'Residual reuse endpoint differs'
        assert self.base._parameter_state_sha256(runtime['parameters'])==self.screen_guard['parameter_state_sha256_after_cast']
        rows=[r for r in self.rows if r['phase']==phase]
        assert len(rows)==4 and all(r['beta']==beta and r['example_count']==len(source_indices) for r in rows)
        assert phase not in self.seen
        self.seen.add(phase)
    def summary_rows(self,*,context,expected_examples,p90_threshold):
        assert self.seen=={'post_T_free','zero','negative','positive'}
        for row in self.rows:
            assert all(row[key]==value for key,value in context.items()), 'Residual reference context differs'
            assert row['expected_example_count']==expected_examples and row['gate_threshold']==p90_threshold
        return [dict(row) for row in self.rows]
    def archive_records(self,*,context):
        # The current runner publishes the existing summary rows, not synthetic
        # per-example raw records. Their original hashes remain in those rows.
        return []


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


def verify_screen_reference(base,output,screen_guard,screen_rows,batch_index,*,allow_bptt_hash_mismatch=False):
    """Require identical physical states; record any authorized BPTT hash waiver."""
    bptt_matches=[]
    for key in ('common_post_T_state_sha256','positive_state_sha256',
                'negative_state_sha256','zero_state_sha256','frozen_current_force_sha256',
                'parameter_state_sha256_after_cast','reconstructed_initialization_tensor_sha256'):
        assert output['guard'][key] == screen_guard[key], f'Screen reconstruction differs: {key}'
    for name,gradient in output['bptt_gradients'].items():
        row = screen_rows[(batch_index,name)]
        matches=base._tensor_sha256(gradient) == row['bptt_gradient_sha256']
        bptt_matches.append(matches)
        if not allow_bptt_hash_mismatch:
            assert matches, 'Screen BPTT differs'
        clean = output['gradients'][name]
        assert math.isclose(float(clean.square().mean().sqrt()),float(row['clean_eqprop_rms']),rel_tol=1e-12,abs_tol=1e-15), 'Screen clean EqProp differs'
    return all(bptt_matches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', choices=('cuda',), default='cuda')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--target')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    bptt_validation=cfg.get('bptt_reference_validation','exact_screen_hash')
    assert bptt_validation in ('exact_screen_hash','unchecked_cross_device_user_override')
    allow_bptt_hash_mismatch=bptt_validation=='unchecked_cross_device_user_override'
    out = args.output_root.resolve() if args.output_root else ROOT / cfg['output_root']
    spec = importlib.util.spec_from_file_location(
        'init_noise_gate', ROOT / cfg['analyzer_source_root'] /
        'analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    base, shadow, torch = gate.base, gate.shadow, gate.torch
    allocator_cap,total_device_memory=configure_cuda(torch,cfg)
    memory_records=[]
    def release_cache(stage):
        if cfg['release_unused_cuda_cache']:
            gc.collect(); torch.cuda.empty_cache()
        memory_records.append(dict(stage=stage,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()))
    from experiments.reporting import (
        append_metric, atomic_write_json, complete_run, fail_run,
        start_run, update_status_progress, validate_run,
    )

    def verify_inputs():
        assert base.sha256_file(ROOT/cfg['parent_config']['path']) == cfg['parent_config']['sha256']
        parent=json.loads((ROOT/cfg['parent_config']['path']).read_text())
        for case in cfg['cases']:
            source=next(c for c in parent['cases'] if c['name']==case['name'])
            assert all(case[k]==v for k,v in source.items()), 'Parent scientific case changed'
        for name, expected in cfg['analyzer_sha256'].items():
            assert base.sha256_file(ROOT / cfg['analyzer_source_root'] / name) == expected, name
        cohort_path = ROOT / cfg['cohort_config']
        assert base.sha256_file(cohort_path) == cfg['cohort_config_sha256']
        prior = json.loads(cohort_path.read_text())
        for name, expected in prior['runtime_sha256'].items():
            assert base.sha256_file(Path(cfg['source_contract']['runtime_source_root']) / name) == expected, name
        for case in cfg['cases']:
            assert base.sha256_file(ROOT / case['initializer_checkpoint_path']) == case['initializer_checkpoint_sha256']
            for reference in (case['calibration_proof'],*case['screen_references'].values()):
                assert base.sha256_file(ROOT/reference['path']) == reference['sha256']
            assert base.sha256_file(ROOT / case['source_config']) == case['source_config_sha256']
            screen_manifest=json.loads((ROOT/case['screen_references']['manifest']['path']).read_text())
            assert screen_manifest['configuration']['sha256']==cfg['parent_config']['sha256']
            assert screen_manifest['git']['runner_sha256']==cfg['parent_runner_sha256']
            assert screen_manifest['cohort_sha256']==cfg['expected_cohort_sha256']
            assert all(screen_manifest['inputs'][key]==value for key,value in next(c for c in parent['cases'] if c['name']==case['name']).items())
        return prior

    prior = verify_inputs()
    resolved = {**cfg, 'dataset': copy.deepcopy(prior['dataset'])}
    if args.data_root:
        resolved['dataset']['root'] = str(args.data_root)
    assert len(cfg['cases']) == cfg['expected_case_count'] > 0
    assert len({c['name'] for c in cfg['cases']}) == len(cfg['cases'])
    draws = noise_draws(cfg)
    assert cfg['saved_activation_cpu_offload'] is True and cfg['cuda_required'] is True
    assert cfg['saved_activation_cpu_offload_scope'] == 'physical_replay_and_acquisition'
    assert cfg['residual_diagnostic_policy']=='reuse_hash_pinned_identical_screen_endpoints'
    assert len(cfg['cases']) == 3 and {c['scheme'] for c in cfg['cases']} == {'baseline','ours','legacy'}
    assert len({c['relative_target'] for c in cfg['cases']}) == 1
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
    elif cfg.get('smoke_gate_policy')=='skipped_by_explicit_user_override':
        assert allow_bptt_hash_mismatch
        prior_seconds=float(cfg['previous_failed_target_smoke_seconds'])
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
    original_bptt=shadow._native_bptt_weight_gradients
    def memory_bptt(*args,**kwargs):
        assert cfg['saved_activation_cpu_offload'] is True
        assert_cuda_runtime(args[0])
        # The outer physical-replay context also covers residual autograd.
        result=original_bptt(*args,**kwargs)
        assert_cuda_runtime(args[0])
        release_cache('after_BPTT')
        return result
    cache = BpttCache(memory_bptt,base,verify_cached=False)
    shadow._native_bptt_weight_gradients = cache
    for declared in cfg['cases']:
        check_deadline()
        source_config = json.loads((ROOT / declared['source_config']).read_text())
        gate._bias_lr_proof(source_config)
        source_config['init_checkpoint_path'] = str(ROOT / declared['initializer_checkpoint_path'])
        source_config['initialization'] = {'checkpoint_sha256': declared['initializer_checkpoint_sha256']}
        model = base._model_config(source_config)
        assert declared['architecture']=='conv3' and declared['relative_target'] in (2.,6.)
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
        screen_guards=json.loads((ROOT/declared['screen_references']['guards']['path']).read_text())
        reference_residuals=load_residual_reference(ROOT/declared['screen_references']['residuals']['path'])
        reference_phases=load_phase_reference(ROOT/declared['screen_references']['phases']['path'])
        with (ROOT/declared['screen_references']['layer_metrics']['path']).open(newline='') as stream:
            screen_rows={(int(r['batch_index']),r['parameter_name'].strip()):r for r in csv.DictReader(stream) if float(r['readout_sigma'])==0}
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
                         cuda_version=torch.version.cuda, device=args.device,
                         saved_activation_cpu_offload=cfg['saved_activation_cpu_offload'],pin_memory=True,
                         saved_activation_cpu_offload_scope=cfg['saved_activation_cpu_offload_scope'],
                         residual_diagnostic_policy=cfg['residual_diagnostic_policy'],
                         bptt_reference_validation=bptt_validation,
                         exploratory_cross_device_bptt_unchecked=allow_bptt_hash_mismatch,
                         smoke_gate_policy=cfg.get('smoke_gate_policy','required_pass'),
                         cuda_allocator_cap_bytes=allocator_cap,cuda_device_total_memory_bytes=total_device_memory,
                         release_unused_cuda_cache=cfg['release_unused_cuda_cache'],
                         cuda_allocator_config=os.environ.get('PYTORCH_CUDA_ALLOC_CONF')),
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
                batch_residuals=[r for r in reference_residuals if r['batch_index']==int(batch['batch_index'])]
                batch_phases=[r for r in reference_phases if r['batch_index']==int(batch['batch_index'])]
                assert all(r['batch_payload_sha256']==batch['payload_sha256'] and
                           r['batch_source_indices_sha256']==batch['source_indices_sha256'] for r in batch_phases)
                shadow.base._phase_residual=lambda runtime:reused_phase_residual(
                    runtime,base,batch_phases,screen_guards[int(batch['batch_index'])])
                shadow.extended._ResidualAccumulator=lambda:ReusedResidualAccumulator(
                    batch_residuals,base,screen_guards[int(batch['batch_index'])])
                release_cache('before_physical_replay')
                output = offloaded_call(torch,shadow._run_precision,case=case, selected=selected, batch=batch,
                    precision='float64', device=torch.device(args.device), residual_threshold=.01,
                    amplification_exponent=depth, nudging_mode='current', eqprop_variant='centered',
                    capture_endpoint_states=True)
                release_cache('after_physical_replay_before_acquisition')
                guard = output['guard']
                bptt_bitwise_match=verify_screen_reference(base,output,screen_guards[int(batch['batch_index'])],screen_rows,int(batch['batch_index']),allow_bptt_hash_mismatch=allow_bptt_hash_mismatch)
                guard['screen_bptt_bitwise_match']=bptt_bitwise_match
                guard['bptt_reference_validation']=bptt_validation
                guard['exploratory_cross_device_bptt_unchecked']=allow_bptt_hash_mismatch
                assert guard['parameter_tensors_unchanged_after_cast']
                assert guard['frozen_current_force_matches_post_T_cost_gradient']
                assert guard['frozen_current_force_unchanged_across_phases']
                runtime = output['captured_runtime']
                assert_cuda_runtime(runtime)
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
                        negative_free_rms=rms(negative-free), centered_half_difference_rms=rms((positive-negative)/2),
                        positive_rms=rms(positive),negative_rms=rms(negative),
                        endpoint_rms=math.sqrt((rms(positive)**2+rms(negative)**2)/2),
                        endpoint_rms_P=math.sqrt((rms(positive)**2+rms(negative)**2)/2),
                        D_over_P=rms((positive-negative)/2)/math.sqrt((rms(positive)**2+rms(negative)**2)/2) if rms(positive) or rms(negative) else None))
                assert list(output['gradients']) == declared['parameter_order']
                for noise_model,level,seed,draw_index in draws:
                    sigma=level if noise_model=='absolute' else 0.
                    noise = gate._read_noise_contract({})
                    noise.update(endpoint_read_noise_std=sigma,
                                 endpoint_read_noise_seed=seed)
                    scored,numerators,records,noise_guard=offloaded_call(torch,apply_read_noise,gate,output=output,case=case,batch=batch,
                        model=noise_model,level=level,seed=seed,device=torch.device(args.device))
                    if args.smoke and noise_model=='absolute':
                        reference,reference_numerators,_,reference_guard=offloaded_call(torch,gate._apply_endpoint_read_noise,
                            output=output,case=case,batch=batch,noise_contract=noise,device=torch.device(args.device))
                        assert all(torch.equal(scored[k],reference[k]) and torch.equal(numerators[k],reference_numerators[k]) for k in scored), 'Absolute path differs from frozen estimator'
                        assert reference_guard['parameter_tensors_unchanged'] and reference_guard['input_remained_exact']
                    assert noise_guard['parameter_tensors_unchanged'] and noise_guard['input_remained_exact']
                    for record in records:
                        check_noise_record(noise_hashes,record,architecture=case['architecture'],
                            seed=seed,batch_index=int(batch['batch_index']))
                        noise_records.append(dict(record,injected_beta=case['injected_beta'],
                            relative_target=declared['relative_target'],noise_seed=seed,noise_draw_index=draw_index))
                    output.update(scored_gradients=scored, scored_numerators=numerators, noise_contract=noise)
                    for name, clean in output['gradients'].items():
                        row = gate._layer_row(config=resolved, case=case, selected=selected, batch=batch,
                                              parameter_name=name, output=output, residual_passed=residual_passed)
                        bptt, noisy = output['bptt_gradients'][name], scored[name]
                        noise_norm = float((noisy-clean).norm())
                        clean_norm = float(clean.norm())
                        row.update(checkpoint_epoch=0,readout_sigma=level if noise_model=='absolute' else (0. if noise_model=='clean' else None),
                            endpoint_read_noise_std=level if noise_model=='absolute' else (0. if noise_model=='clean' else None),
                            noise_model=noise_model,noise_level=level,level_units='dimensionless' if noise_model=='nodewise_relative' else 'voltage',
                            noise_draw_index=draw_index,noise_seed=seed,
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
                    release_cache('after_acquisition')
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
            assert len(noise_records) == len(batches)*sum(model!='clean' for model,_,_,_ in draws)*2*(depth+1)
            assert base.sha256_file(run / 'checkpoints/initialization.pt') == declared['initializer_checkpoint_sha256']
            for name, records in [('layer_metrics', rows), ('equilibrium_residuals', residuals),
                                   ('state_signal', signals), ('state_displacement', displacements),
                                   ('phase_diagnostics', phases), ('endpoint_read_noise', noise_records)]:
                if records:
                    base._write_csv(run / f'{name}.csv', records)
            atomic_write_json(run / 'read_only_guards.json', guards)
            atomic_write_json(run / 'cuda_memory.json',memory_records)
            memory_records.clear()
            complete_run(run, terminal_metrics=dict(replay_batches=len(batches), layer_comparisons=len(rows),
                noise_tensor_draws=len(noise_records), source_bytes_unchanged=True,
                all_parameter_tensors_unchanged=True, all_bias_tensors_exact_zero=True,
                checkpoint_epoch=0, checkpoint_role=cfg['checkpoint_role'], official_test_read=False,
                actual_relative_output_rms=actual_ratio, relative_target=declared['relative_target'],
                cuda_peak_memory_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                absolute_estimator_parity_passed=True if args.smoke else None,
                screen_reference_hashes_matched=all(g['screen_bptt_bitwise_match'] for g in guards),
                physical_screen_reference_hashes_matched=True,
                bptt_reference_validation=bptt_validation,
                exploratory_cross_device_bptt_unchecked=allow_bptt_hash_mismatch,
                smoke_gate_policy=cfg.get('smoke_gate_policy','required_pass'),
                saved_activation_cpu_offload=True,
                saved_activation_cpu_offload_scope=cfg['saved_activation_cpu_offload_scope'],
                live_network_and_parameters_on_cuda=True,cuda_allocator_cap_bytes=allocator_cap,
                residuals_reused_from_prior_identical_states=True,
                phase_residuals_reused_from_prior_identical_states=True,
                relative_noise_floor=0.,acquisition_draws_per_batch=len(draws),
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
        noise_draws_matched_across_models_and_levels=True,screen_reference_hashes_matched=True,
        elapsed_seconds=time.monotonic()-started, total_including_smoke_seconds=time.monotonic()-started+prior_seconds,
        official_test_read=False, optimizer_steps_applied=False, accuracy_evaluation=False))
    print('COMPLETE', len(summary), 'cases', round(time.monotonic()-started, 1), 'seconds', flush=True)


if __name__ == '__main__':
    main()
