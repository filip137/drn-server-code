"""Read-only native-T Conv2 BPTT voltage comparison across target encodings."""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.reporting import (append_metric, atomic_write_json, complete_run,
    fail_run, runtime_context, sha256_file, start_run, update_status_progress)
from experiments.replay_clean_eqprop_voltages import (assert_parameter_values,
    load_checkpoint_states, pool_statistics, require, tensor_statistics)

ROOT = Path(__file__).resolve().parents[1]


def run(args):
    cfg = json.loads(args.config.read_text())
    spec = importlib.util.spec_from_file_location('_target_voltage_gate',
        ROOT / cfg['analyzer_source_root'] / 'analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py')
    gate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gate
    spec.loader.exec_module(gate)
    torch, base, shadow = gate.torch, gate.base, gate.shadow
    torch.set_num_threads(1)
    out = args.output_root.resolve()
    cases = [c for c in cfg['cases'] if not args.smoke or c['seed'] == 0]
    started = time.monotonic()
    runtime_root = ROOT / cfg['source_contract']['runtime_source_root']
    code_hashes = {str(p.relative_to(runtime_root)): sha256_file(p)
                  for d in ('model', 'labs', 'training') for p in sorted((runtime_root / d).rglob('*.py'))}
    provenance = dict(runner_sha256=sha256_file(Path(__file__)), runtime_source_sha256=code_hashes,
        analyzer_source_sha256={p.name: sha256_file(p) for p in sorted((ROOT / cfg['analyzer_source_root']).glob('*.py'))},
        statistics_helper_sha256=sha256_file(ROOT / 'experiments/replay_clean_eqprop_voltages.py'),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    def manifest(name, case=None):
        return dict(study_id=cfg['study_id'], run_id=name, arm_id=name, seed=case['seed'] if case else 0,
            evidence_class='ordinary_mnist_selection', dataset='ordinary_mnist_validation',
            diagnostic='finite_iteration_voltage_replay', smoke=args.smoke,
            command=[sys.executable, *sys.argv], configuration=cfg,
            config_sha256=sha256_file(args.config), inputs=case or {}, git=provenance,
            runtime={**runtime_context(target='local_cpu'), 'torch':torch.__version__, 'threads':1},
            official_test_read=False, optimizer_steps_applied=False)
    start_run(out, manifest('voltage-replay'))
    current = None
    all_profiles = []
    try:
        source0 = json.loads((ROOT / cases[0]['source_config']).read_text())
        batches, cohort = gate._build_cohort(cfg, [{'source_config':source0}])
        require(cohort['cohort_sha256'] == cfg['expected_cohort_sha256'], 'Cohort changed')
        atomic_write_json(out / 'cohort.json', cohort)
        batches = batches[:1] if args.smoke else batches
        for case in cases:
            current = out / 'cases' / case['name']
            start_run(current, manifest(case['name'], case))
            source_path, checkpoint = ROOT / case['source_config'], ROOT / case['checkpoint_path']
            require(sha256_file(source_path) == case['source_config_sha256'], 'Source config changed')
            require(sha256_file(checkpoint) == case['checkpoint_sha256'], 'Checkpoint changed')
            source = json.loads(source_path.read_text())
            model = base._model_config(source)
            require(source['training_algorithm'] == 'BP' and source['optimizer']['name'].lower() == 'adam', 'Not BPTT Adam')
            require(source['lab']['epochs'] == 30 and source['seed'] == case['seed'], 'Training contract changed')
            require(model['num_iterations_inference'] == model['num_iterations_training'] == 6, 'T/K mismatch')
            require(model['input_gain'] == 100 and model['non_linearity'] == 'perfect_diode', 'Model mismatch')
            require(model['weight_min'] == case['weight_min'] and model['weight_max'] == case['weight_max'], 'Bounds differ')
            require((model['voltage_amp'],model['current_amp']) == {'baseline':(1,1),'ours':(4,1),'legacy':(4,.25)}[case['scheme']], 'Scheme differs')
            require(model['layer_shapes'] == [[2,28,28],[64,14,14],[128,7,7],[20]], 'Architecture differs')
            require(source['energy_minimizer']['mode'] == 'asynchronous' and source['batch_state_policy'] == 'reset_each_batch', 'State policy differs')
            params = source['datasets']['mnist']['params']
            require(params['split_seed'] == 0 and params['normalize'] and not params.get('affine_config'), 'Dataset differs')
            for k,v in [('normalize_mean',.1307),('normalize_std',.3081),('normalize_scale',.3)]:
                require(params[k] == v, f'Preprocessing differs: {k}')
            require(source['target_scale'] == case['target_scale'], 'Target encoding differs')
            source = copy.deepcopy(source)
            source.pop('init_checkpoint_path', None)
            runtime = base._build_runtime(source, device=torch.device('cpu'), gradient_iterations=6)
            runtime['energy_fn'].load(checkpoint)
            expected = load_checkpoint_states(torch, checkpoint)
            require(all(v.dtype == torch.float32 for v in expected), 'Expected native float32 checkpoint')
            assert_parameter_values(runtime['parameters'], expected)
            for p in runtime['parameters']:
                if 'Weight' in p.name:
                    require(bool((p.state >= model['weight_min']).all()) and bool((p.state <= model['weight_max']).all()), 'Out-of-bounds weights')
            before = base._parameter_state_sha256(runtime['parameters'])
            rows = []
            for batch in batches:
                require(time.monotonic()-started < (180 if args.smoke else cfg['budget_seconds']), 'Time budget exhausted')
                with torch.no_grad():
                    shadow._set_input_and_reset(runtime, batch['images'].to(dtype=torch.float32), dtype=torch.float32)
                    runtime['minimizer_inference'].compute_equilibrium()
                    for index, layer in enumerate(runtime['free_layers']):
                        rows.append(dict(batch_index=batch['batch_index'], layer_index=index, **tensor_statistics(layer.state)))
                progress = dict(stage='voltage_replay', case=case['name'], batch_index=batch['batch_index'],
                    completed_cases=len(all_profiles)//3, elapsed_seconds=time.monotonic()-started, dataset_split='validation')
                for path in (current, out):
                    append_metric(path/'metrics.jsonl', progress)
                    update_status_progress(path, progress)
            assert_parameter_values(runtime['parameters'], expected)
            require(before == base._parameter_state_sha256(runtime['parameters']), 'Parameters changed')
            require(sha256_file(checkpoint) == case['checkpoint_sha256'] and sha256_file(source_path) == case['source_config_sha256'], 'Input bytes changed')
            context = {k:case[k] for k in ('name','scheme','seed','weight_min','weight_max','target_scale','checkpoint_epoch','checkpoint_path')}
            profiles = [dict(**context, layer_index=i, layer=['Conv1','Conv2','Readout'][i],
                **pool_statistics([r for r in rows if r['layer_index']==i])) for i in range(3)]
            require(len(rows) == 3*len(batches), 'Incomplete layers/batches')
            base._write_csv(current/'voltage_sufficient_statistics.csv', rows)
            base._write_csv(current/'voltage_profiles.csv', profiles)
            guards = dict(exact_restore=True, parameters_unchanged=True, source_bytes_unchanged=True,
                finite_states=True, zero_biases=True, native_float32=True, examples=len(batches)*16,
                cohort_sha256=cohort['cohort_sha256'], checkpoint_epoch=case['checkpoint_epoch'])
            atomic_write_json(current/'read_only_guards.json', guards)
            complete_run(current, terminal_metrics=guards)
            all_profiles.extend(profiles)
            current = None
            print('DONE', case['name'], 'elapsed', round(time.monotonic()-started,2), flush=True)
            del runtime, expected
            gc.collect()
        base._write_csv(out/'voltage_profiles.csv', all_profiles)
        summary = dict(cases=len(cases), profile_rows=len(all_profiles), examples_per_case=len(batches)*16,
            elapsed_seconds=time.monotonic()-started, full_cohort=not args.smoke, official_test_read=False,
            checkpoint_role='final', T=6, K=6, runtime_dtype='float32')
        atomic_write_json(out/'summary.json', summary)
        complete_run(out, terminal_metrics=summary)
        return summary
    except BaseException as error:
        if current is not None:
            fail_run(current,error=error)
        fail_run(out,error=error)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    print(run(parser.parse_args()))
