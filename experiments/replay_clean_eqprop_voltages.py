"""Read-only, voltage-only replay of clean EqProp initial/final checkpoints.

Paths in the JSON config are relative to the repository root. The frozen
analyzer's source_contract.runtime_source_root is also required for its import.
Smoke processes the first batch of the same guarded 576-example cohort; the
full-cohort initialization reference comparison is deliberately deferred.
"""
from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

from experiments.reporting import (
    append_metric, atomic_write_json, complete_run, fail_run, runtime_context,
    sha256_file, start_run, update_status_progress,
)

ROOT = Path(__file__).resolve().parents[1]
AMPLIFICATIONS = {'baseline': (1., 1.), 'ours': (4., 1.), 'legacy': (4., .25)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensor_statistics(value):
    """Sufficient statistics, retaining counts rather than averaging batch RMS."""
    import torch
    value = value.detach().to(dtype=torch.float64)
    if not value.numel() or not bool(torch.isfinite(value).all()):
        raise FloatingPointError('Empty or nonfinite voltage tensor')
    return dict(element_count=value.numel(), voltage_sum=float(value.sum()),
                absolute_sum=float(value.abs().sum()), square_sum=float(value.square().sum()),
                minimum=float(value.min()), maximum=float(value.max()),
                zero_count=int((value == 0).sum()))


def pool_statistics(records):
    require(bool(records), 'Cannot pool an empty voltage profile')
    n = sum(r['element_count'] for r in records)
    require(n > 0, 'Voltage profile has no elements')
    mean = math.fsum(r['voltage_sum'] for r in records) / n
    square_mean = math.fsum(r['square_sum'] for r in records) / n
    return dict(element_count=n, signed_mean=mean,
                mean_absolute=math.fsum(r['absolute_sum'] for r in records) / n,
                rms=math.sqrt(square_mean), std=math.sqrt(max(0., square_mean - mean * mean)),
                minimum=min(r['minimum'] for r in records), maximum=max(r['maximum'] for r in records),
                zero_count=sum(r['zero_count'] for r in records),
                zero_fraction=sum(r['zero_count'] for r in records) / n)


def check_initial_profile(observed, expected):
    require(len(observed) == len(expected), 'Initialization reference layer count differs')
    for layer, (actual, reference) in enumerate(zip(observed, expected)):
        require(math.isfinite(actual) and math.isfinite(reference)
                and math.isclose(actual, reference, rel_tol=1e-10, abs_tol=1e-12),
                f'Initialization free RMS differs at layer {layer}: {actual} != {reference}')


def assert_parameter_values(parameters, expected):
    import torch
    require(len(parameters) == len(expected), 'Checkpoint parameter count differs')
    for parameter, saved in zip(parameters, expected):
        current = parameter.state.detach().cpu()
        reference = saved.detach().cpu().to(dtype=current.dtype)
        require(current.shape == reference.shape and torch.equal(current, reference),
                f'Checkpoint parameter changed: {parameter.name}')
        require(bool(torch.isfinite(current).all()), f'Nonfinite parameter: {parameter.name}')
        if str(parameter.name).strip().startswith('Bias_'):
            require(bool((current == 0).all()), f'Nonzero frozen bias: {parameter.name}')


def load_checkpoint_states(torch, path):
    # Historical frozen runtimes use tensor lists; modern checkpoints carry an
    # explicit states array. The runtime load still validates schema.
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(payload, dict):
        payload = payload['states']
    require(isinstance(payload, (list, tuple)) and all(torch.is_tensor(v) for v in payload),
            f'Unsupported model checkpoint: {path}')
    return payload


def validate_case(base, declared, source):
    depth = int(declared['architecture'][-1])
    T, gain = {1: (4, 40), 2: (6, 100), 3: (8, 360)}[depth]
    model = base._model_config(source)
    require(declared['T'] == declared['K'] == model['num_iterations_inference']
            == model['num_iterations_training'] == T, 'Native T/K mismatch')
    require(declared['input_gain'] == model['input_gain'] == gain, 'Input gain mismatch')
    require((model['voltage_amp'], model['current_amp']) == AMPLIFICATIONS[declared['scheme']],
            'Amplification scheme mismatch')
    require(model['weight_min'] == 0 and model['weight_max'] == 100, 'Weight bounds mismatch')
    require(model['non_linearity'] == 'perfect_diode', 'Expected perfect diodes')
    require(source['seed'] == 0 and source['training_algorithm'] == 'EP'
            and source['runtime_dtype'] == 'float64', 'Expected seed-0 float64 EqProp training')
    require(source.get('batch_state_policy') == 'reset_each_batch', 'Unexpected batch state policy')
    ep = source['eqprop']
    require(ep['nudging_mode'] == 'current' and ep['variant'] == 'centered', 'Nudging contract differs')
    require(ep['endpoint_read_noise_eta'] == ep['endpoint_read_noise_std'] == 0
            and not ep['input_read_noise'] and not ep['validation_read_noise'], 'Expected clean checkpoint')
    require(declared['injected_beta'] == ep['injected_beta_B']
            and math.isfinite(declared['injected_beta']) and declared['injected_beta'] > 0,
            'Saved training injected beta differs')
    role = declared['checkpoint_role']
    require(role in ('initialization', 'final'), 'Unknown checkpoint role')
    require(declared['checkpoint_epoch'] == (0 if role == 'initialization' else (10 if depth == 1 else 30)),
            'Checkpoint epoch differs from frozen contract')
    return depth


def replay_batch(gate, runtime, batch, injected_beta, device):
    """Only T free updates and two centered K endpoints; no parameter gradients."""
    torch, base, shadow = gate.torch, gate.base, gate.shadow
    with torch.no_grad():
        images = batch['images'].to(device=device, dtype=torch.float64)
        shadow._set_input_and_reset(runtime, images, dtype=torch.float64)
        runtime['minimizer_inference'].compute_equilibrium()
        runtime['cost_fn'].set_target(batch['labels'].to(device=device))
        free = base._clone_states(runtime['free_layers'])
        for value in free:
            base._finite(value, 'post-T free voltage')
        runtime['augmented_fn'].prepare_nudging()
        force = runtime['augmented_fn']._nudging._force.detach().clone()
        expected_force = -runtime['cost_fn']._grad(runtime['cost_fn'].layers()[-1], mean=False)
        require(torch.equal(force, expected_force), 'Frozen force differs from post-T cost derivative')
        states = {}
        for phase, sign in (('positive', 1), ('negative', -1)):
            base._restore_states(runtime['free_layers'], free)
            require(all(torch.equal(layer.state, value) for layer, value in zip(runtime['free_layers'], free)),
                    'Centered endpoints did not share the post-T start')
            runtime['augmented_fn'].nudging = torch.tensor(
                sign * injected_beta / runtime['nudging_current_scale'], device=device, dtype=torch.float64)
            runtime['minimizer_augmented'].compute_equilibrium()
            require(torch.equal(force, runtime['augmented_fn']._nudging._force), 'Frozen current force changed')
            states[phase] = base._clone_states(runtime['free_layers'])
        rows = []
        for index, free_value in enumerate(free):
            for phase, value in (('free', free_value), ('positive', states['positive'][index]),
                                 ('negative', states['negative'][index]),
                                 ('centered_half_difference', (states['positive'][index] - states['negative'][index]) / 2)):
                rows.append(dict(batch_index=int(batch['batch_index']), state_layer_index=index,
                                 state_layer_name=f'Layer_{index + 1}', phase=phase, **tensor_statistics(value)))
    return rows


def aggregate_profiles(rows, depth):
    profiles = []
    for index in range(depth + 1):
        phases = {}
        for phase in ('free', 'positive', 'negative', 'centered_half_difference'):
            phases[phase] = pool_statistics([r for r in rows if r['state_layer_index'] == index and r['phase'] == phase])
        F = phases['free']['rms']
        D = phases['centered_half_difference']['rms']
        P = math.sqrt((phases['positive']['rms'] ** 2 + phases['negative']['rms'] ** 2) / 2)
        profiles.append(dict(state_layer_index=index, state_layer_name=f'Layer_{index + 1}',
            **{f'{phase}_{key}': value for phase, stats in phases.items() for key, value in stats.items()},
            endpoint_rms_P=P, centered_half_difference_rms_D=D,
            D_over_F=D / F if F else None, D_over_P=D / P if P else None))
    return profiles


def run(args):
    cfg = json.loads(args.config.read_text())
    out = (args.output_root or ROOT / cfg['output_root']).resolve()
    path = ROOT / cfg['analyzer_source_root'] / 'analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py'
    spec = importlib.util.spec_from_file_location('_clean_voltage_gate', path)
    gate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gate
    spec.loader.exec_module(gate)
    torch, base, shadow = gate.torch, gate.base, gate.shadow
    device = torch.device(args.device)
    if device.type == 'cuda':
        require(torch.cuda.is_available(), 'CUDA was requested but is unavailable')
        cap = cfg['cuda_allocator_cap_bytes']
        total = torch.cuda.get_device_properties(0).total_memory
        require(0 < cap < total, 'Invalid CUDA allocator cap')
        torch.cuda.set_per_process_memory_fraction(cap / total, device=0)
    prior = json.loads((ROOT / cfg['cohort_config']).read_text())
    resolved = {**cfg, 'dataset': copy.deepcopy(prior['dataset'])}
    if args.data_root:
        resolved['dataset']['root'] = str(args.data_root.resolve())
    require(resolved['dataset']['batch_size'] == 16 and resolved['dataset']['example_count'] == 576,
            'Expected 576 validation examples in batches of 16')
    started = time.monotonic()
    config_hash = sha256_file(args.config)
    source_hashes = {str(p.relative_to(ROOT / cfg['source_contract']['runtime_source_root'])): sha256_file(p)
                     for directory in ('model', 'labs')
                     for p in sorted((ROOT / cfg['source_contract']['runtime_source_root'] / directory).rglob('*.py'))}
    analyzer_hashes = {p.name: sha256_file(p) for p in sorted(path.parent.glob('*.py'))}
    def manifest(name, inputs=None):
        return dict(study_id=cfg['study_id'], run_id=name, arm_id=name, seed=0,
                    evidence_class='ordinary_mnist_selection', dataset='ordinary_mnist_validation',
                    diagnostic='finite_iteration_voltage_replay', smoke=args.smoke,
                    command=[sys.executable, *sys.argv], configuration=dict(path=str(args.config.resolve()), sha256=config_hash, resolved=resolved),
                    inputs=inputs or {}, git=dict(runner_sha256=sha256_file(Path(__file__)),
                        frozen_runtime_source_sha256=source_hashes, analyzer_source_sha256=analyzer_hashes),
                    runtime=dict(runtime_context(target=args.target or cfg.get('target')), device=str(device),
                                 torch_version=torch.__version__, cuda_version=torch.version.cuda,
                                 cuda_allocator_cap_bytes=cfg['cuda_allocator_cap_bytes']),
                    official_test_read=False, optimizer_steps_applied=False,
                    gradient_diagnostics_run=False, residual_diagnostics_run=False, beta_recalibrated=False)
    start_run(out, manifest('voltage-replay'))
    atomic_write_json(out / 'config.used.json', cfg)
    atomic_write_json(out / 'resolved_config.json', resolved)
    current_run = None
    completed, all_profiles = [], []
    def progress(stage, name, batch_index=None):
        elapsed = time.monotonic() - started
        row = dict(stage=stage, dataset_split='validation', case=name, batch_index=batch_index,
                   completed_cases=len(completed), expected_cases=len(cfg['cases']), elapsed_seconds=elapsed)
        for bundle in (out, current_run):
            if bundle is not None:
                append_metric(bundle / 'metrics.jsonl', row)
                update_status_progress(bundle, row)
        if elapsed >= cfg['budget_seconds']:
            raise TimeoutError('Voltage replay budget exhausted')
    try:
        names = [c['name'] for c in cfg['cases']]
        require(len(names) == len(set(names)), 'Duplicate case names')
        require(all(Path(n).name == n and n not in ('.', '..') for n in names), 'Unsafe case name')
        dataset_source = json.loads((ROOT / cfg['cases'][0]['source_config']).read_text())
        batches, cohort = gate._build_cohort(resolved, [{'source_config': dataset_source}])
        require(len(batches) == 36 and cohort['cohort_sha256'] == cfg['expected_cohort_sha256'], 'Cohort identity differs')
        atomic_write_json(out / 'cohort.json', cohort)
        batches = batches[:1] if args.smoke else batches
        for case in cfg['cases']:
            current_run = out / 'cases' / case['name']
            start_run(current_run, manifest(case['name'], case))
            progress('restore', case['name'])
            expected_hashes = {ROOT / case[pathkey]: case[hashkey] for pathkey, hashkey in (
                ('source_config', 'source_config_sha256'), ('checkpoint_path', 'checkpoint_sha256'),
                ('initialization_checkpoint_path', 'initialization_checkpoint_sha256'))}
            for input_path, digest in expected_hashes.items():
                require(sha256_file(input_path) == digest, f'Input hash mismatch: {input_path}')
            source = json.loads((ROOT / case['source_config']).read_text())
            depth = validate_case(base, case, source)
            dataset_options = dict(data_root=Path(resolved['dataset']['root']).resolve(), batch_size=16)
            require(base._dataset_signature(source, **dataset_options)
                    == base._dataset_signature(dataset_source, **dataset_options),
                    'Case dataset/preprocessing differs from the frozen cohort')
            if case['checkpoint_role'] == 'initialization':
                require(case['checkpoint_sha256'] == case['initialization_checkpoint_sha256'],
                        'Initialization replay does not use its declared initial checkpoint')
            # Loading a native float64 checkpoint through float32 would round it.
            source.pop('init_checkpoint_path', None)
            runtime = base._build_runtime(source, device=device, gradient_iterations=case['K'], nudging_mode='current', current_scale='auto')
            shadow._convert_runtime_dtype(runtime, torch.float64)
            runtime['energy_fn'].load(ROOT / case['checkpoint_path'])
            expected = load_checkpoint_states(torch, ROOT / case['checkpoint_path'])
            require(all(p.state.dtype == torch.float64 and p.state.device.type == device.type for p in runtime['parameters']),
                    'Parameter precision or device mismatch')
            assert_parameter_values(runtime['parameters'], expected)
            reference_hash = case.get('reference_initialization_tensor_sha256')
            if reference_hash and case['checkpoint_role'] == 'initialization':
                require(base._historical_initialization_sha256(runtime['parameters']) == reference_hash,
                        'Historical initialization tensor identity differs')
            initial_parameter_hash = base._parameter_state_sha256(runtime['parameters'])
            expected_scale = (AMPLIFICATIONS[case['scheme']][0] / AMPLIFICATIONS[case['scheme']][1]) ** depth
            require(runtime['nudging_current_scale'] == expected_scale, 'Frozen-current scale differs')
            rows = []
            for batch in batches:
                progress('voltage_replay', case['name'], int(batch['batch_index']))
                rows.extend(replay_batch(gate, runtime, batch, case['injected_beta'], device))
                assert_parameter_values(runtime['parameters'], expected)
            profiles = aggregate_profiles(rows, depth)
            initial_reference_checked = case['checkpoint_role'] == 'initialization' and not args.smoke
            if initial_reference_checked:
                check_initial_profile([r['free_rms'] for r in profiles], case['expected_initial_free_rms'])
            require(len(rows) == len(batches) * (depth + 1) * 4, 'Incomplete voltage coverage')
            for input_path, digest in expected_hashes.items():
                require(sha256_file(input_path) == digest, f'Input changed during replay: {input_path}')
            require(base._parameter_state_sha256(runtime['parameters']) == initial_parameter_hash, 'Parameter bytes changed')
            context = {k: case[k] for k in ('name', 'architecture', 'scheme', 'checkpoint_role', 'checkpoint_epoch', 'injected_beta', 'T', 'K', 'input_gain')}
            profiles = [dict(context, **r) for r in profiles]
            base._write_csv(current_run / 'voltage_sufficient_statistics.csv', [dict(context, **r) for r in rows])
            base._write_csv(current_run / 'voltage_profiles.csv', profiles)
            guard = dict(checkpoint_restore_exact=True, parameters_unchanged=True,
                         parameter_sha256=initial_parameter_hash, source_checkpoint_bytes_unchanged=True,
                         source_config_bytes_unchanged=True, zero_biases=True, frozen_current_force_unchanged=True,
                         centered_endpoints_share_post_T_start=True, initialization_reference_checked=initial_reference_checked,
                         initialization_reference_deferred_for_smoke=case['checkpoint_role'] == 'initialization' and args.smoke,
                         cohort_sha256=cohort['cohort_sha256'], native_float64=True)
            atomic_write_json(current_run / 'read_only_guards.json', guard)
            complete_run(current_run, terminal_metrics=dict(batches=len(batches), examples=len(batches) * 16,
                layers=depth + 1, **guard, peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device.type == 'cuda' else 0))
            completed.append(case['name'])
            all_profiles.extend(profiles)
            current_run = None
            progress('case_complete', case['name'])
            print('DONE', case['name'], 'elapsed', round(time.monotonic() - started, 2), flush=True)
            del runtime, expected, rows
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        base._write_csv(out / 'voltage_profiles.csv', all_profiles)
        summary = dict(cases=completed, expected_cases=len(cfg['cases']), examples_per_case=len(batches) * 16,
                       profile_rows=len(all_profiles), elapsed_seconds=time.monotonic() - started,
                       cohort_sha256=cohort['cohort_sha256'], full_cohort=not args.smoke,
                       official_test_read=False, optimizer_steps_applied=False, beta_recalibrated=False)
        atomic_write_json(out / 'summary.json', summary)
        complete_run(out, terminal_metrics=summary)
        return summary
    except BaseException as error:
        if current_run is not None:
            fail_run(current_run, error=error)
        fail_run(out, error=error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--target')
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
