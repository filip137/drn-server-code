"""Prepare exp025 rescaled-loss targets and rematch all 12 contexts on accepted576.

Run prepare once, then calibrate each depth/scheme on its assigned device. Calibration
uses the current numerical source, accepted float32 tensors loaded before the
float64 conversion (the exact-run initialization order), and frozen clean force.
No training, optimizer search, or GPU placement is performed by this script.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.calibrate_initial_relative_output import matched_beta, pooled_ratio, sha
from experiments.exact_run import load_exact_config
from experiments.reporting import (
    append_metric, atomic_write_json, complete_run, fail_run, start_run,
    update_status_progress, validate_run,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'conv123-rescaled-target-relative-noise-jeanzay-20261001-v1'
OUT = Path('results') / STUDY
CONFIGS = Path('configs/conv/rescaled_target_noise_20261001')
COHORT = Path('configs/conv/eqprop_conv3_p90_final_noise_cosine_20260921.json')
TARGETS = {1: 1, 2: 4, 3: 6}
ETAS = ((0., 'eta0'), (1e-6, 'eta1em6'), (1e-5, 'eta1em5'), (1e-4, 'eta1em4'))
SCHEMES = ('baseline', 'ours', 'legacy', 'amp2')
TRAINING_DATA_ROOT = '/lustre/fsn1/projects/rech/umg/ucy17uy/datasets/mnist'


def write_once(path, value):
    path = ROOT / path
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise FileExistsError(f'Refusing to overwrite changed input: {path}')
    else:
        atomic_write_json(path, value)


def source_identity():
    paths = set()
    for directory in ('model', 'training', 'labs'):
        paths.update(p for p in (ROOT / directory).rglob('*.py')
                     if 'tests' not in p.relative_to(ROOT).parts)
    paths.update(ROOT / 'experiments' / name for name in (
        'prepare_rescaled_target_noise.py', 'prepare_current_amp2.py', 'calibrate_initial_relative_output.py',
        'analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py',
        'audit_eqprop_float64_shadow.py', 'analyze_conv_eqprop_bptt_beta_tk_displacement.py',
        'analyze_conv_eqprop_bptt_checkpoint_gradients.py', 'exact_run.py', 'reporting.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def training_config(source, depth, scheme, eta, label, initializer, calibration=None):
    """Change the actual target; keep inherited LR except the declared amp2 factor."""
    from experiments.prepare_current_amp2 import training_config as current_amp2_config
    c = current_amp2_config(source, depth, scheme, eta, label, initializer)
    c.pop('current_amp2_comparison')
    c.pop('relative_output_calibration', None)
    c.pop('relative_noise_eta_extension', None)
    arm = f'conv{depth}_{scheme}_target{4**depth}_{label}_seed0'
    c.update(study_id=STUDY, arm_id=arm, target_scale=4.**(-depth))
    c['datasets']['mnist']['params']['root'] = TRAINING_DATA_ROOT
    c['reporting'].update(study_id=STUDY, arm_id=arm)
    scale = (c['model_base']['voltage_amp']/c['model_base']['current_amp'])**depth
    injected_beta = source['eqprop']['injected_beta_B'] if calibration is None else calibration['injected_beta']
    c['beta'] = injected_beta/scale
    c['eqprop'].update(amplification_factor=scale, injected_beta_B=injected_beta,
                      endpoint_read_noise_seed=2026081601)
    c['rescaled_target_noise'] = dict(scheme=scheme, relative_target=TARGETS[depth],
        target_scale=c['target_scale'], eta=eta, calibration_reused=False,
        learning_rate_multiplier_of_source=2 if scheme == 'amp2' else 1,
        primary_metric=f'epoch_{c["lab"]["epochs"]}_validation_accuracy',
        uninterrupted_optimizer_history=True, beta_frozen=True,
        checkpoint_recovery='full_state_epoch_boundary_only',
        actual_loss='0.5 * sum((paired_scores - target_scale * one_hot)**2)',
        loss_normalization=False, learning_rate_target_compensation=False)
    if calibration is not None:
        c['relative_output_calibration'] = copy.deepcopy(calibration)
    assert c['training_algorithm'] == 'EP'
    assert c['datasets']['mnist']['params']['batch_size'] == 16
    assert c['datasets']['mnist']['params']['validation_batch_size'] == 64
    assert c['model_base']['num_iterations_inference'] == 2*depth+2
    assert c['model_base']['num_iterations_training'] == 2*depth+2
    assert c['evaluation']['official_test']['policy'] == 'disabled'
    return c


def cohort_digest(source):
    calibration = source['relative_output_calibration']
    if 'measured' in calibration:
        return calibration['measured']['cohort_sha256']
    return calibration['seed0_reference']['measured']['cohort_sha256']


def prepare(args):
    if not (ROOT/args.assigned_experiment).is_file():
        raise FileNotFoundError(args.assigned_experiment)
    provenance = dict(study_id=STUDY, source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256=source_identity(), inputs={}, calibration_configs=[])
    cohort = json.loads((ROOT / COHORT).read_text())['dataset']
    for depth, target in TARGETS.items():
        initializer_source = ROOT / ('results/conv1-clean-and-baseline-noise-20260927-v1' if depth == 1 else
                                     'results/conv23-noiseless-three-seed-20260927-v1') / f'source-training-v1/inputs/initializers/conv{depth}_seed0.pt'
        initializer = OUT / f'inputs/initializers/conv{depth}_seed0.pt'
        (ROOT/initializer).parent.mkdir(parents=True, exist_ok=True)
        if not (ROOT/initializer).exists():
            shutil.copyfile(initializer_source, ROOT/initializer)
        for scheme in SCHEMES:
            source_scheme = 'ours' if scheme == 'amp2' else scheme
            src = Path(f'configs/conv/relative_noise_eta5em4_20260928/conv{depth}_{source_scheme}_relative{target}_eta5em4_seed0.json')
            source = json.loads((ROOT/src).read_text())
            if sha(ROOT/initializer) != source['initialization']['checkpoint_sha256'] or sha(initializer_source) != sha(ROOT/initializer):
                raise ValueError(f'Initializer identity mismatch: {initializer}')
            inputs = {str(src):sha(ROOT/src), str(initializer):sha(ROOT/initializer), str(COHORT):sha(ROOT/COHORT)}
            provenance['inputs'].update(inputs)
            template = training_config(source, depth, scheme, 0., 'eta0', initializer)
            template['rescaled_target_noise'].update(source_config=str(src), source_config_sha256=sha(ROOT/src))
            cal = dict(study_id=STUDY, architecture=f'conv{depth}', depth=depth, scheme=scheme,
                target=target, target_scale=4.**(-depth), output_root=str(OUT),
                training_config_root=str(CONFIGS), source_contract={'runtime_source_root': '.'}, dataset=cohort,
                source_commit=provenance['source_commit'], source_sha256=provenance['source_sha256'],
                input_sha256=inputs, assigned_experiment=args.assigned_experiment,
                template=template, pilot_injected_beta=source['eqprop']['injected_beta_B'],
                relative_tolerance=.01, max_corrections=3, calibration_budget_seconds=1800,
                expected_cohort_sha256=cohort_digest(source))
            path = CONFIGS / f'provenance/calibration_conv{depth}_{scheme}.json'
            write_once(path, cal)
            provenance['calibration_configs'].append(dict(path=str(path), sha256=sha(ROOT/path)))
    write_once(CONFIGS / 'provenance/preparation.json', provenance)
    print(json.dumps(dict(training_configs=0, calibration_configs=12, expected_training_configs=48,
                         study_root=str(OUT), source_files=len(provenance['source_sha256']))))


def verify(cfg):
    for path, digest in {**cfg['source_sha256'], **cfg['input_sha256']}.items():
        if sha(ROOT/path) != digest:
            raise ValueError(f'Source/input identity changed: {path}')


def load_gate():
    spec = importlib.util.spec_from_file_location('rescaled_target_gate', ROOT/'experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    return gate


def calibrate(args):
    cfg = json.loads(args.config.read_text())
    verify(cfg)
    if not (ROOT/cfg['assigned_experiment']).is_file():
        raise FileNotFoundError(cfg['assigned_experiment'])
    gate = load_gate()
    base, shadow, torch = gate.base, gate.shadow, gate.torch
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError('Accepted initializer schema requires the default float32 constructor.')
    depth = cfg['depth']
    study_root = args.output_root or ROOT/cfg['output_root']
    out = study_root/'calibration'/(('smoke_' if args.smoke else '')+cfg['architecture']+'_'+cfg['scheme'])
    if out.exists():
        raise FileExistsError(out)
    start = time.monotonic()
    start_run(out, dict(study_id=STUDY, run_id=out.name, arm_id=out.name,
        evidence_class='ordinary_mnist_selection', dataset='ordinary_mnist_validation', smoke=args.smoke,
        command=[sys.executable, *sys.argv], configuration=dict(path=str(args.config),sha256=sha(args.config),resolved=cfg),
        git=dict(commit=cfg['source_commit'], dirty_worktree=True, source_sha256=cfg['source_sha256']),
        runtime=dict(base.runtime_context(target=args.target),device=args.device,torch_version=torch.__version__),
        official_test_read=False, optimizer_steps_applied=False))
    try:
        source = copy.deepcopy(cfg['template'])
        source['init_checkpoint_path'] = str(ROOT/source['init_checkpoint_path'])
        cohort_cfg = copy.deepcopy(cfg)
        if args.data_root:
            cohort_cfg['dataset']['root'] = str(args.data_root.resolve())
        batches, cohort = gate._build_cohort(cohort_cfg, [{'source_config':source}])
        assert len(batches) == 36 and cohort['cohort_sha256'] == cfg['expected_cohort_sha256']
        atomic_write_json(out/'cohort.json', cohort)
        if args.smoke:
            batches = batches[:1]
        runtime = base._build_runtime(source, device=torch.device(args.device), gradient_iterations=2*depth+2,
                                      nudging_mode='current', current_scale='auto')
        shadow._convert_runtime_dtype(runtime, torch.float64)
        saved = torch.load(source['init_checkpoint_path'], map_location=args.device, weights_only=False)
        parameters = runtime['parameters']
        assert len(saved['states']) == len(parameters)
        for parameter, accepted in zip(parameters, saved['states']):
            assert parameter.state.dtype == torch.float64
            assert torch.equal(parameter.state, accepted.to(dtype=torch.float64))
            if str(parameter.name).startswith('Bias_'):
                assert not bool(parameter.state.count_nonzero())
        initial_hash = base._parameter_state_sha256(parameters)
        scale = (source['model_base']['voltage_amp']/source['model_base']['current_amp'])**depth
        assert runtime['cost_fn']._target_scale == cfg['target_scale']
        assert math.isclose(runtime['nudging_current_scale'], scale, rel_tol=1e-12)
        measurements, cache = [], {}

        def measure(beta):
            if beta in cache:
                return cache[beta]
            rows = []
            for batch in batches:
                if time.monotonic()-start > cfg['calibration_budget_seconds']:
                    raise TimeoutError('Calibration budget exhausted')
                shadow._set_input_and_reset(runtime, batch['images'].to(device=args.device,dtype=torch.float64),dtype=torch.float64)
                runtime['minimizer_inference'].compute_equilibrium()
                runtime['cost_fn'].set_target(batch['labels'].to(args.device))
                states = base._clone_states(runtime['free_layers'])
                output = runtime['cost_fn'].layers()[-1]
                assert tuple(output.shape) == (20,)
                force = -runtime['cost_fn']._grad(output, mean=False).detach().clone()
                runtime['augmented_fn'].prepare_nudging()
                assert torch.equal(runtime['augmented_fn']._nudging._force, force)
                endpoints = []
                for sign in (-1, 1):
                    base._restore_states(runtime['free_layers'], states)
                    runtime['augmented_fn'].nudging = torch.as_tensor(sign*beta/scale,dtype=torch.float64,device=args.device)
                    runtime['minimizer_augmented'].compute_equilibrium()
                    assert all(bool(torch.isfinite(x.state).all()) for x in runtime['free_layers'])
                    endpoints.append(runtime['free_layers'][-1].state.detach().clone())
                    assert torch.equal(runtime['augmented_fn']._nudging._force, force)
                centered = (endpoints[1]-endpoints[0])/2
                rows.append(dict(batch_index=int(batch['batch_index']), elements=states[-1].numel(),
                    free_sum_squares=float(states[-1].square().sum()), centered_sum_squares=float(centered.square().sum())))
                update_status_progress(out,dict(stage='calibration', injected_beta=beta,batches=len(rows)))
            assert initial_hash == base._parameter_state_sha256(parameters)
            value = pooled_ratio(rows)
            cache[beta] = value
            measurements.append(dict(injected_beta=beta,base_beta=beta/scale,batches=rows,**value))
            atomic_write_json(out/'measurements.json',measurements)
            append_metric(out/'metrics.jsonl',dict(stage='calibration',dataset_split='validation',injected_beta=beta,**value))
            print('MEASURE', cfg['architecture'], beta, value['relative_output_rms'],flush=True)
            return value

        pilot = measure(cfg['pilot_injected_beta'])
        beta, history = matched_beta(measure, cfg['pilot_injected_beta']*cfg['target']/pilot['relative_output_rms'],
            cfg['target'], cfg['relative_tolerance'], cfg['max_corrections'])
        calibration = dict(target=cfg['target'], definition='D/F: pooled physical output half-difference RMS / post-T free RMS over 576x20 nodes',
            T=2*depth+2,K=2*depth+2,injected_beta=beta,base_beta=beta/scale,measured=history[-1],history=history,
            cohort_sha256=cohort['cohort_sha256'],matching_examples=16 if args.smoke else 576,
            initializer_parameter_state_sha256=initial_hash, all_parameters_unchanged=True,
            source_config=str(args.config.relative_to(ROOT) if args.config.is_absolute() else args.config),source_config_sha256=sha(args.config),
            target_scale=cfg['target_scale'], actual_loss_target=True, beta_frozen=True)
        atomic_write_json(out/'matching.json',calibration)
        emitted = []
        verify(cfg)
        if not args.smoke:
            # Rebuild from the original identified config; amp2 rates are doubled exactly once.
            original = json.loads((ROOT/cfg['template']['rescaled_target_noise']['source_config']).read_text())
            for eta,label in ETAS:
                c = training_config(original,depth,cfg['scheme'],eta,label,Path(cfg['template']['init_checkpoint_path']),calibration)
                c['rescaled_target_noise'].update(cfg['template']['rescaled_target_noise'],eta=eta)
                path = Path(cfg['training_config_root'])/(c['arm_id']+'.json')
                write_once(path,c)
                load_exact_config(ROOT/path)
                emitted.append(dict(path=str(path),sha256=sha(ROOT/path)))
        atomic_write_json(out/'training_configs.json',emitted)
        verify(cfg)
        complete_run(out,terminal_metrics=dict(matches=1,relative_target_error=history[-1]['relative_target_error'],
            injected_beta=beta, base_beta=beta/scale, official_test_read=False, optimizer_steps_applied=False,
            elapsed_seconds=time.monotonic()-start),completion=dict(criteria_met=True,coverage_complete=True))
        assert not validate_run(out)
    except BaseException as error:
        fail_run(out,error=error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','calibrate'))
    parser.add_argument('--assigned-experiment')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--target', default='local')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output-root', type=Path, help='Study result root; defaults to the prepared study root.')
    args = parser.parse_args()
    if args.action == 'prepare':
        if not args.assigned_experiment:
            parser.error('prepare requires --assigned-experiment')
        prepare(args)
    else:
        if not args.config:
            parser.error('calibrate requires --config')
        calibrate(args)


if __name__ == '__main__':
    main()
