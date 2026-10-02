"""Exploratory matched-displacement map with global relative endpoint read noise."""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time

from experiments.replay_relative_endpoint_noise import (
    BpttCache, apply_read_noise, assert_cuda_runtime, check_noise_record,
    configure_cuda, offloaded_call,
)
from experiments.reporting import (
    append_metric, atomic_write_json, complete_run, fail_run,
    start_run, update_status_progress,
)

ROOT = Path(__file__).resolve().parents[1]
AMPLIFICATIONS = {
    'baseline': (1., 1.), 'ours': (4., 1.), 'legacy': (4., .25),
    'amp2': (4., 2.),
    'v1_c0p5': (1., .5), 'v1_c0p25': (1., .25),
}


class NoResidualDiagnostics:
    """No residual qualification was requested; empty results are not passes."""
    def add_endpoint(self, *args, **kwargs): pass
    def summary_rows(self, **kwargs): return []
    def archive_records(self, **kwargs): return []


def cosine(a, b):
    denominator=float(a.norm())*float(b.norm())
    return float((a.flatten()*b.flatten()).sum())/denominator if denominator else None


def layer_rows(output, scored, *, context, model, eta, seed, draw):
    rows=[]
    for name,clean in output['gradients'].items():
        bptt=output['bptt_gradients'][name];gradient=scored[name]
        difference=gradient-clean
        bnorm=float(bptt.norm());cnorm=float(clean.norm())
        rows.append(dict(context,parameter_name=name,state_layer_index=list(output['gradients']).index(name),
            noise_model=model,noise_level=eta,noise_draw_index=draw,noise_seed=seed,
            cosine=cosine(gradient,bptt),noisy_clean_eqprop_cosine=cosine(gradient,clean),
            clean_eqprop_bptt_cosine=cosine(clean,bptt),
            bptt_rms=float(bptt.square().mean().sqrt()),clean_eqprop_rms=float(clean.square().mean().sqrt()),
            eqprop_rms=float(gradient.square().mean().sqrt()),
            added_noise_rms=float(difference.square().mean().sqrt()),
            noise_over_clean_norm=float(difference.norm())/cnorm if cnorm else None,
            relative_gradient_error_norm=float((gradient-bptt).norm())/bnorm if bnorm else None))
    return rows


def state_rows(output, *, context):
    rows=[]
    rms=lambda value:float(value.square().mean().sqrt())
    for layer,free in enumerate(output['captured_post_t_states']):
        positive=output['captured_phase_states']['positive'][layer]
        negative=output['captured_phase_states']['negative'][layer]
        F=rms(free);D=rms((positive-negative)/2)
        P=math.sqrt((rms(positive)**2+rms(negative)**2)/2)
        rows.append(dict(context,state_layer_index=layer,state_layer_name=f'Layer_{layer+1}',element_count=free.numel(),
            free_rms=F,positive_rms=rms(positive),negative_rms=rms(negative),endpoint_rms=P,endpoint_rms_P=P,
            centered_half_difference_rms=D,D_over_F=D/F if F else None,D_over_P=D/P if P else None))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--device',choices=['cuda'],default='cuda')
    parser.add_argument('--target')
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='One batch per case through the same replay and noise path.')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    out=args.output_root.resolve() if args.output_root else ROOT/cfg['output_root']
    spec=importlib.util.spec_from_file_location('relative_advantage_gate',ROOT/cfg['analyzer_source_root']/'analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py')
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
    base,shadow,torch=gate.base,gate.shadow,gate.torch
    configure_cuda(torch,cfg)
    device=torch.device('cuda')
    shadow.extended._ResidualAccumulator=NoResidualDiagnostics
    shadow.base._phase_residual=lambda runtime:{}
    prior=json.loads((ROOT/cfg['cohort_config']).read_text())
    resolved={**cfg,'dataset':copy.deepcopy(prior['dataset'])}
    if args.data_root:resolved['dataset']['root']=str(args.data_root)
    assert cfg['stage'] in ('screen','refinement','layerwise')
    assert len(cfg['read_noise_seeds'])==len(set(cfg['read_noise_seeds']))==3
    assert cfg['noisy_batches'] in (12,36) and cfg['max_matching_corrections']==2
    assert cfg['target_relative_tolerance']==.01 and cfg['expected_batches']==36
    assert not cfg['official_test_read'] and not cfg['optimizer_steps_applied']
    clean_batches = 1 if args.smoke else cfg['expected_batches']
    noisy_batches = 1 if args.smoke else cfg['noisy_batches']
    config_hash=base.sha256_file(args.config)
    started=time.monotonic();resume_spent=0.
    if (out/'status.json').exists():
        if not args.resume:raise FileExistsError(out)
        old_manifest=json.loads((out/'manifest.json').read_text())
        assert old_manifest['configuration']['sha256']==config_hash,'Resume config differs'
        old_status=json.loads((out/'status.json').read_text())
        if old_status['state']=='complete':return
        if old_status['state']=='running':raise RuntimeError('Resolve the prior running process before resuming')
        resume_spent=float(old_status.get('progress',{}).get('elapsed_seconds',0.))
        metrics_path=out/'metrics.jsonl'
        if metrics_path.exists():
            for line in metrics_path.read_text().splitlines():
                resume_spent=max(resume_spent,float(json.loads(line).get('elapsed_seconds',0.)))
        archive=out/'attempts'/f'lane-{time.time_ns()}';archive.mkdir(parents=True)
        for name in ('manifest.json','status.json','metrics.jsonl','result.json'):
            if (out/name).exists():shutil.move(str(out/name),archive/name)
    def elapsed():return time.monotonic()-started+resume_spent
    def deadline():
        if elapsed()+cfg.get('prior_gpu_seconds',0.)>=cfg['budget_seconds']:
            raise TimeoutError('Lane budget across stages exhausted')
    def manifest(name,inputs=None):
        return dict(study_id=cfg['study_id'],run_id=name,arm_id=name,evidence_class='ordinary_mnist_selection',
            dataset='ordinary_mnist_validation',smoke=args.smoke,command=[sys.executable,*sys.argv],
            configuration=dict(path=str(args.config),sha256=config_hash,resolved=resolved),inputs=inputs or {},
            git=dict(commit=cfg.get('source_commit'),runner_sha256=base.sha256_file(Path(__file__))),
            runtime=dict(base.runtime_context(target=args.target or cfg['target']),device='cuda',torch_version=torch.__version__,
                cuda_version=torch.version.cuda,saved_activation_cpu_offload=True,cuda_allocator_cap_bytes=cfg['cuda_allocator_cap_bytes']),
            official_test_read=False,optimizer_steps_applied=False,exploratory=True,
            tests_and_canaries_skipped_by_user=cfg.get('tests_and_canaries_skipped_by_user',False),
            historical_cross_device_equivalence_checked=False,residual_diagnostics_run=False)
    start_run(out,manifest(f"{cfg['stage']}-{cfg['architecture']}"))
    atomic_write_json(out/'config.used.json',cfg);atomic_write_json(out/'resolved_config.json',resolved)
    current_run=None;completed=[];noise_hashes={}
    def progress(stage,case_name,batch=None,correction=None):
        value=dict(stage=stage,case=case_name,batch_index=batch,correction=correction,
                   completed_cases=len(completed),expected_cases=len(cfg['cases']),elapsed_seconds=elapsed())
        append_metric(out/'metrics.jsonl',value);update_status_progress(out,value)
        if current_run is not None:
            append_metric(current_run/'metrics.jsonl',value);update_status_progress(current_run,value)
    try:
        dataset_source=json.loads((ROOT/cfg['cases'][0]['source_config']).read_text())
        batches,cohort=gate._build_cohort(resolved,[{'source_config':dataset_source}])
        assert len(batches)==36 and cohort['cohort_sha256']==cfg['expected_cohort_sha256']
        atomic_write_json(out/'cohort.json',cohort)
        batches = batches[:clean_batches]
        cache=BpttCache(shadow._native_bptt_weight_gradients,base,verify_cached=False)
        shadow._native_bptt_weight_gradients=cache
        for declared in cfg['cases']:
            deadline()
            current_run=out/'cases'/declared['name']
            if (current_run/'result.json').exists():
                assert args.resume
                old=json.loads((current_run/'manifest.json').read_text())
                assert old['configuration']['sha256']==config_hash
                completed.append(declared['name']);continue
            if current_run.exists():
                assert args.resume
                archive=out/'attempts'/f"{declared['name']}-{time.time_ns()}"
                archive.parent.mkdir(exist_ok=True);shutil.move(str(current_run),archive)
            start_run(current_run,manifest(declared['name'],declared))
            for path_key, hash_key in (('source_config','source_config_sha256'),
                                       ('initializer_checkpoint_path','initializer_checkpoint_sha256')):
                assert base.sha256_file(ROOT/declared[path_key]) == declared[hash_key], f'Input changed: {path_key}'
            source=json.loads((ROOT/declared['source_config']).read_text())
            source['init_checkpoint_path']=str(ROOT/declared['initializer_checkpoint_path'])
            source['initialization']={'checkpoint_sha256':declared['initializer_checkpoint_sha256']}
            model=base._model_config(source)
            depth=int(declared['architecture'][-1]);T,gain={1:(4,40),2:(6,100),3:(8,360)}[depth]
            assert model['num_iterations_inference']==model['num_iterations_training']==T
            assert model['input_gain']==gain and model['weight_min']==0 and model['weight_max']==100
            voltage,current=float(model['voltage_amp']),float(model['current_amp'])
            assert (voltage,current)==AMPLIFICATIONS[declared['scheme']]
            r=float(declared['relative_target']);levels=declared.get('noise_levels',cfg['noise_levels'])
            assert levels and all(math.isfinite(x) and x>0 for x in levels)
            B=float(declared['injected_beta'])*r/float(declared.get('beta_reference_target',r))
            case=dict(architecture=declared['architecture'],scheme=declared['scheme'],T=T,K=T,native_T=T,native_K=T,
                voltage_amp=voltage,current_amp=current,voltage_amplification=voltage,current_amplification=current,
                source_config=source,reconstructed_initialization_tensor_sha256=declared['reconstructed_initialization_tensor_sha256'])
            history=[];runtime=None
            for correction in range(cfg['max_matching_corrections']+1):
                case.update(injected_beta=B,base_beta=B/(voltage/current)**depth)
                selected=gate._selected(case,cfg['checkpoint_role'])
                clean_rows=[];signals=[];endpoint_cache=[];guards=[]
                for batch in batches:
                    deadline();index=int(batch['batch_index'])
                    cache.context=(case['architecture'],case['scheme'],index,batch['payload_sha256'])
                    gc.collect();torch.cuda.empty_cache()
                    output=offloaded_call(torch,shadow._run_precision,case=case,selected=selected,batch=batch,
                        precision='float64',device=device,residual_threshold=.01,amplification_exponent=depth,
                        nudging_mode='current',eqprop_variant='centered',capture_endpoint_states=True)
                    runtime=output['captured_runtime'];assert_cuda_runtime(runtime)
                    assert all(bool((p.state==0).all()) for p in runtime['parameters'] if str(p.name).startswith('Bias_'))
                    context=dict(stage=cfg['stage'],architecture=case['architecture'],scheme=case['scheme'],relative_target=r,
                        injected_beta=B,base_beta=case['base_beta'],batch_index=index)
                    clean_rows.extend(layer_rows(output,output['gradients'],context=context,model='clean',eta=0.,seed=None,draw=-1))
                    signals.extend(state_rows(output,context=context));guards.append(output['guard'])
                    if index<noisy_batches:
                        stored={key:output[key] for key in ('gradients','numerators','endpoint_gradients','guard','captured_phase_states','captured_input_state','bptt_gradients')}
                        stored['captured_force']=runtime['augmented_fn']._nudging._force.detach().cpu().clone()
                        endpoint_cache.append(stored)
                    del output
                    progress('matching_clean',declared['name'],index,correction)
                output_signals=[row for row in signals if row['state_layer_index']==depth]
                count=sum(row['element_count'] for row in output_signals)
                F=math.sqrt(sum(row['element_count']*row['free_rms']**2 for row in output_signals)/count)
                D=math.sqrt(sum(row['element_count']*row['centered_half_difference_rms']**2 for row in output_signals)/count)
                actual=D/F;error=abs(actual/r-1)
                assert math.isfinite(actual) and actual>0
                history.append(dict(correction=correction,injected_beta=B,base_beta=case['base_beta'],free_rms=F,
                    centered_rms=D,actual_relative_output_rms=actual,relative_error=error))
                matched=error<=cfg['target_relative_tolerance']
                if matched or correction==cfg['max_matching_corrections']:break
                B*=r/actual
            matching=dict(relative_target=r,actual_relative_output_rms=actual,injected_beta=B,base_beta=case['base_beta'],
                matched=matched,free_rms=F,centered_rms=D,relative_error=error,corrections=correction,history=history,
                cohort_sha256=cohort['cohort_sha256'],matching_examples=16*clean_batches)
            atomic_write_json(current_run/'matching.json',matching)
            atomic_write_json(current_run/'read_only_guards.json',guards)
            if cfg['stage']=='layerwise' and not matched:
                raise RuntimeError(f'Output displacement matching failed: {declared["name"]}, error={error}')
            rows=clean_rows;noise_records=[]
            for index,output in enumerate(endpoint_cache):
                batch=batches[index];output['captured_runtime']=runtime
                runtime['augmented_fn']._nudging._force=output['captured_force'].to(device=device)
                context=dict(stage=cfg['stage'],architecture=case['architecture'],scheme=case['scheme'],relative_target=r,
                    injected_beta=B,base_beta=case['base_beta'],batch_index=int(batch['batch_index']))
                for eta in levels:
                    for draw,seed in enumerate(cfg['read_noise_seeds']):
                        deadline()
                        scored,_,records,_=offloaded_call(torch,apply_read_noise,gate,output=output,case=case,batch=batch,
                            model='nodewise_relative',level=eta,seed=seed,device=device)
                        rows.extend(layer_rows(output,scored,context=context,model='nodewise_relative',eta=eta,seed=seed,draw=draw))
                        for record in records:
                            check_noise_record(noise_hashes,record,architecture=case['architecture'],seed=seed,batch_index=index)
                            noise_records.append(dict(record,stage=cfg['stage'],relative_target=r,injected_beta=B,
                                noise_seed=seed,noise_draw_index=draw))
                output.pop('captured_runtime')
                progress('relative_noise',declared['name'],index,correction)
            for collection in (rows,signals,noise_records):
                for row in collection:row.update(matched=matched,actual_relative_output_rms=actual)
            expected=(depth+1)*(clean_batches+noisy_batches*len(levels)*len(cfg['read_noise_seeds']))
            assert len(rows)==expected
            base._write_csv(current_run/'layer_metrics.csv',rows)
            base._write_csv(current_run/'state_signal.csv',signals)
            base._write_csv(current_run/'endpoint_read_noise.csv',noise_records)
            complete_run(current_run,terminal_metrics=dict(layer_comparisons=len(rows),matched=matched,
                actual_relative_output_rms=actual,relative_target=r,injected_beta=B,clean_batches=clean_batches,noisy_batches=noisy_batches,
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                residual_diagnostics_run=False,official_test_read=False,optimizer_steps_applied=False))
            completed.append(declared['name'])
            current_run=None
            progress('case_complete',declared['name'])
            print('DONE',declared['name'],'matched',matched,'ratio',actual,'rows',len(rows),'elapsed',round(elapsed(),2),flush=True)
            current_run=None
            del runtime,endpoint_cache,rows,noise_records;gc.collect();torch.cuda.empty_cache()
        summary=dict(stage=cfg['stage'],architecture=cfg['architecture'],state='complete',cases=completed,elapsed_seconds=elapsed(),
            prior_gpu_seconds=cfg.get('prior_gpu_seconds',0.),budget_seconds=cfg['budget_seconds'],
            bptt_computations=cache.computations,bptt_cache_hits=cache.hits,expected_layer_comparisons=sum((int(c['architecture'][-1])+1)*(clean_batches+noisy_batches*len(c.get('noise_levels',cfg['noise_levels']))*len(cfg['read_noise_seeds'])) for c in cfg['cases']))
        atomic_write_json(out/'summary.json',summary)
        complete_run(out,terminal_metrics=summary,completion=dict(criteria_met=True,coverage_complete=True))
    except BaseException as exc:
        if current_run is not None and not (current_run/'result.json').exists():fail_run(current_run,error=exc)
        append_metric(out/'metrics.jsonl',dict(stage='failed',elapsed_seconds=elapsed(),completed_cases=len(completed)))
        fail_run(out,error=exc)
        raise


if __name__=='__main__':
    main()
