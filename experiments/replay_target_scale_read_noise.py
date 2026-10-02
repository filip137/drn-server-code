"""Read-only relative endpoint-noise replay of rescaled-target BPTT checkpoints."""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from experiments.reporting import (append_metric, atomic_write_json, complete_run,
    fail_run, runtime_context, sha256_file, start_run, update_status_progress)
from experiments.replay_relative_noise_advantage import NoResidualDiagnostics, layer_rows, state_rows
from experiments.replay_relative_endpoint_noise import (BpttCache, apply_read_noise,
    check_noise_record, configure_cuda, offloaded_call)
from experiments.replay_clean_eqprop_voltages import assert_parameter_values, load_checkpoint_states

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--prior-seconds', type=float, default=0)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    from experiments import analyze_conv123_zero_bias_eqprop_bptt_gradient_gate as gate
    base, shadow, torch = gate.base, gate.shadow, gate.torch
    torch.set_num_threads(1)
    configure_cuda(torch, {'cuda_allocator_cap_bytes': int(7.5 * 1024**3)})
    shadow.extended._ResidualAccumulator = NoResidualDiagnostics
    shadow.base._phase_residual = lambda runtime: {}
    device = torch.device('cuda')
    out = args.output_root.resolve()
    started = time.monotonic()
    current = None
    noise_hashes = {}
    provenance = {str(p.relative_to(ROOT)): sha256_file(p)
                  for folder in ('model', 'training', 'labs')
                  for p in sorted((ROOT / folder).rglob('*.py'))}
    for name in ('replay_target_scale_read_noise', 'replay_relative_endpoint_noise',
                 'replay_relative_noise_advantage', 'audit_eqprop_float64_shadow',
                 'analyze_conv123_zero_bias_eqprop_bptt_gradient_gate',
                 'analyze_conv_eqprop_bptt_checkpoint_gradients',
                 'analyze_conv_eqprop_bptt_beta_tk_displacement'):
        p = ROOT / 'experiments' / (name + '.py')
        provenance[str(p.relative_to(ROOT))] = sha256_file(p)

    def manifest(name, case=None):
        return dict(study_id=cfg['study_id'], run_id=name, arm_id=name, seed=0,
            evidence_class='ordinary_mnist_selection', dataset='ordinary_mnist_validation',
            command=[sys.executable, *sys.argv], configuration=cfg, inputs=case or {},
            git=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                     source_sha256=provenance, dirty_worktree=True),
            runtime={**runtime_context(target='local'), 'torch':torch.__version__},
            smoke=args.smoke, official_test_read=False, optimizer_steps_applied=False,
            checkpoint_role='best_validation', runtime_dtype='float64', residual_qualification=False)

    def checkpoint_runtime(case, *, role, device, gradient_iterations, nudging_mode):
        assert role == 'best_validation'
        source = dict(case['source_config'])
        source.pop('init_checkpoint_path', None)
        runtime = base._build_runtime(source, device=device,
            gradient_iterations=gradient_iterations, nudging_mode=nudging_mode, current_scale='auto')
        # The historical analyzer constructs unit targets; explicitly preserve
        # the training loss encoding before either BPTT or the frozen EP force.
        assert hasattr(runtime['cost_fn'], '_target_scale')
        runtime['cost_fn']._target_scale = float(source['target_scale'])
        path = case['best_checkpoint_path']
        runtime['energy_fn'].load(path)
        expected = load_checkpoint_states(torch, path)
        assert all(v.dtype == torch.float32 for v in expected)
        assert_parameter_values(runtime['parameters'], expected)
        assert all(bool((p.state == 0).all()) for p in runtime['parameters'] if p.name.startswith('Bias_'))
        return runtime, dict(native_checkpoint_dtype='float32', checkpoint_role=role,
            best_checkpoint_path=str(path), best_checkpoint_sha256=sha256_file(path),
            loaded_checkpoint_exactly=True, target_scale=source['target_scale'],
            native_parameter_state_sha256_before_cast=base._parameter_state_sha256(runtime['parameters']),
            native_parameter_numeric_sha256_before_cast=shadow._numeric_parameter_sha256(runtime['parameters']))

    shadow._checkpoint_runtime = checkpoint_runtime
    cache = BpttCache(shadow._native_bptt_weight_gradients, base, verify_cached=False)
    shadow._native_bptt_weight_gradients = cache
    start_run(out, manifest('replay'))
    atomic_write_json(out/'config.used.json', cfg)
    completed = []

    def progress(stage, name, batch=None):
        elapsed = time.monotonic() - started
        if elapsed + args.prior_seconds >= cfg['budget_seconds']:
            raise TimeoutError('Study elapsed budget exhausted')
        row = dict(stage=stage, case=name, batch_index=batch,
                   completed_cases=len(completed), elapsed_seconds=elapsed)
        for path in (out, current):
            if path is not None:
                append_metric(path/'metrics.jsonl', row)
                update_status_progress(path, row)

    try:
        source0 = json.loads((ROOT/cfg['cases'][0]['bundle']/'config.used.json').read_text())
        batches, cohort = gate._build_cohort(cfg, [{'source_config':source0}])
        assert cohort['cohort_sha256'] == cfg['expected_cohort_sha256']
        batches = batches[:1 if args.smoke else cfg['batch_count']]
        atomic_write_json(out/'cohort.json', dict(cohort, replayed_batches=[b['batch_index'] for b in batches]))
        cases = cfg['cases']
        if args.smoke:
            architectures = dict.fromkeys(c['architecture'] for c in cases)
            cases = [next(c for c in cases if c['architecture']==a) for a in architectures]
        for declared in cases:
            current = out/'cases'/declared['name']
            start_run(current, manifest(declared['name'], declared))
            bundle = ROOT/declared['bundle']
            for name, digest in declared['source_hashes'].items():
                assert sha256_file(bundle/name) == digest, name
            source = json.loads((bundle/'config.used.json').read_text())
            assert source['target_scale'] == declared['target_scale']
            assert source['training_algorithm'] == 'BP'
            model = base._model_config(source)
            T, K = declared['T'], declared['K']
            assert model['num_iterations_inference'] == T and model['num_iterations_training'] == K
            voltage, current_amp = model['voltage_amp'], model['current_amp']
            assert (voltage, current_amp) == {'ours':(4,1), 'legacy':(4,.25), 'amp2':(4,2)}[declared['scheme']]
            depth = int(declared['architecture'][-1])
            case = dict(declared, source_config=source, native_T=T, native_K=K,
                voltage_amp=voltage, current_amp=current_amp,
                voltage_amplification=voltage, current_amplification=current_amp,
                best_checkpoint_path=bundle/'best_model.pt')
            beta = 1e-3
            history = []
            for correction in range(cfg['max_matching_corrections']+1):
                case.update(injected_beta=beta, base_beta=beta/(voltage/current_amp)**depth)
                selected = gate._selected(case, 'best_validation')
                endpoints, signals, rows, guards = [], [], [], []
                for batch in batches:
                    progress('matching', declared['name'], batch['batch_index'])
                    cache.context = (declared['name'], batch['batch_index'], batch['payload_sha256'])
                    gc.collect(); torch.cuda.empty_cache()
                    output = offloaded_call(torch, shadow._run_precision, case=case,
                        selected=selected, batch=batch, precision='float64', device=device,
                        residual_threshold=.01, amplification_exponent=depth,
                        nudging_mode='current', eqprop_variant='centered', capture_endpoint_states=True)
                    runtime = output['captured_runtime']
                    assert float(runtime['cost_fn']._target.max()) == declared['target_scale']
                    context = {k:declared[k] for k in ('name','architecture','scheme','target_scale','checkpoint_epoch','relative_target')}
                    context.update(injected_beta=beta, batch_index=batch['batch_index'])
                    rows.extend(layer_rows(output, output['gradients'], context=context, model='clean',eta=0.,seed=None,draw=-1))
                    signals.extend(state_rows(output, context=context))
                    guards.append(output['guard'])
                    stored = {key:output[key] for key in ('gradients','numerators','endpoint_gradients','guard','captured_phase_states','captured_input_state','bptt_gradients')}
                    stored['captured_force'] = runtime['augmented_fn']._nudging._force.detach().cpu().clone()
                    endpoints.append(stored)
                    del output
                output_signals = [r for r in signals if r['state_layer_index']==depth]
                F2 = sum(r['element_count']*r['free_rms']**2 for r in output_signals)
                D2 = sum(r['element_count']*r['centered_half_difference_rms']**2 for r in output_signals)
                actual = math.sqrt(D2/F2)
                error = abs(actual/declared['relative_target']-1)
                history.append(dict(correction=correction,injected_beta=beta,actual_relative_output_rms=actual,relative_error=error))
                print('MATCH',declared['name'],history[-1],flush=True)
                if error <= cfg['matching_tolerance']:
                    break
                beta *= declared['relative_target']/actual
            atomic_write_json(current/'matching.json', dict(history=history,matched=error<=cfg['matching_tolerance']))
            assert error <= cfg['matching_tolerance'], 'Output D/F did not match'
            records = []
            levels = [cfg['noise_levels'][0], cfg['noise_levels'][-1]] if args.smoke else cfg['noise_levels']
            for batch, output in zip(batches, endpoints):
                output['captured_runtime'] = runtime
                runtime['augmented_fn']._nudging._force = output['captured_force'].to(device)
                context = {k:declared[k] for k in ('name','architecture','scheme','target_scale','checkpoint_epoch','relative_target')}
                context.update(injected_beta=case['injected_beta'], batch_index=batch['batch_index'])
                for eta in levels:
                    for draw, seed in enumerate(cfg['read_noise_seeds']):
                        progress('noise', declared['name'], batch['batch_index'])
                        scored, _, observations, _ = offloaded_call(torch, apply_read_noise, gate,
                            output=output, case=case, batch=batch, model='nodewise_relative',level=eta,seed=seed,device=device)
                        rows.extend(layer_rows(output,scored,context=context,model='nodewise_relative',eta=eta,seed=seed,draw=draw))
                        for record in observations:
                            check_noise_record(noise_hashes,record,architecture=case['architecture'],seed=seed,batch_index=batch['batch_index'])
                        records.extend(observations)
                output.pop('captured_runtime')
            assert len(rows)==(depth+1)*len(batches)*(1+len(levels)*len(cfg['read_noise_seeds']))
            assert all(r['cosine'] is not None and math.isfinite(r['cosine']) for r in rows)
            assert_parameter_values(runtime['parameters'],load_checkpoint_states(torch,bundle/'best_model.pt'))
            for name,digest in declared['source_hashes'].items():
                assert sha256_file(bundle/name)==digest
            base._write_csv(current/'layer_metrics.csv',rows)
            base._write_csv(current/'state_metrics.csv',signals)
            base._write_csv(current/'noise_metrics.csv',records)
            atomic_write_json(current/'read_only_guards.json', guards)
            terminal=dict(layer_rows=len(rows),examples=16*len(batches),actual_relative_output_rms=actual,
                          injected_beta=case['injected_beta'],target_scale=declared['target_scale'],
                          checkpoint_unchanged=True,parameters_unchanged=True,paired_noise=True)
            complete_run(current,terminal_metrics=terminal)
            completed.append(dict(name=declared['name'],**terminal))
            print('DONE',declared['name'],'elapsed',time.monotonic()-started,flush=True)
            current=None
            del runtime,endpoints,rows,records
            gc.collect();torch.cuda.empty_cache()
        atomic_write_json(out/'summary.json',dict(cases=completed,elapsed_seconds=time.monotonic()-started))
        complete_run(out,terminal_metrics=dict(completed_cases=len(completed),expected_cases=len(cases)))
    except BaseException as error:
        if current is not None: fail_run(current,error=error)
        fail_run(out,error=error)
        raise


if __name__ == '__main__':
    main()
