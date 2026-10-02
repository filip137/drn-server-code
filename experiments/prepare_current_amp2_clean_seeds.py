"""Prepare six clean A4/B2 replicates using accepted initializers and seed0 beta.

CPU only: no calibration, relaxation, training, or GPU placement. Run from the
repository root with python -m experiments.prepare_current_amp2_clean_seeds.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

from experiments.exact_run import load_exact_config
from experiments.prepare_current_amp2 import load_gate, source_identity, write_once
from experiments.reporting import sha256_file as sha

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'conv123-current-amp2-clean-seeds12-20260928-v1'
OUT = Path('results') / STUDY
CONFIGS = Path('configs/conv/current_amp2_clean_seeds12_20260928')
TARGETS = {1: 1, 2: 4, 3: 6}


def main():
    # The replay helper otherwise selects a historical archive at import time.
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], '--runtime-source-root', str(ROOT)]
        gate = load_gate()
    finally:
        sys.argv = original_argv
    torch, base, shadow = gate.torch, gate.base, gate.shadow
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError('Accepted initializer loading requires float32 construction')
    provenance = dict(study_id=STUDY, source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256=source_identity(), inputs={}, configs=[], initializer_checks=[])
    for name in ('prepare_current_amp2_clean_seeds.py', 'analyze_current_amp2_clean_seeds.py'):
        path = Path('experiments') / name
        provenance['source_sha256'][str(path)] = sha(ROOT/path)
    for depth, target in TARGETS.items():
        parent_path = Path(f'configs/conv/current_amp2_20260928/conv{depth}_amp2_relative{target}_eta0_seed0.json')
        parent = load_exact_config(ROOT/parent_path)
        provenance['inputs'][str(parent_path)] = sha(ROOT/parent_path)
        historical = Path('results') / ('conv1-clean-and-baseline-noise-20260927-v1' if depth == 1
                                       else 'conv23-noiseless-three-seed-20260927-v1')
        source_config_dir = 'conv1_clean_baseline_noise_20260927' if depth == 1 else 'noiseless_three_seed_20260927'
        hashes = [parent['relative_output_calibration']['initializer_parameter_state_sha256']]
        for seed in (1, 2):
            accepted = historical/f'source-training-v1/inputs/initializers/conv{depth}_seed{seed}.pt'
            accepted_config = historical/f'source-training-v1/configs/conv/{source_config_dir}/conv{depth}_ours_relative{target}_eta0_seed{seed}.json'
            accepted_identity = json.loads((ROOT/accepted_config).read_text())['initialization']['checkpoint_sha256']
            if sha(ROOT/accepted) != accepted_identity:
                raise ValueError(f'Accepted initializer changed: {accepted}')
            initializer = OUT/f'inputs/initializers/conv{depth}_seed{seed}.pt'
            (ROOT/initializer).parent.mkdir(parents=True, exist_ok=True)
            if not (ROOT/initializer).exists():
                shutil.copyfile(ROOT/accepted, ROOT/initializer)
            if sha(ROOT/initializer) != accepted_identity:
                raise ValueError(f'Staged initializer changed: {initializer}')
            for path in (accepted, accepted_config, initializer):
                provenance['inputs'][str(path)] = sha(ROOT/path)
            config = copy.deepcopy(parent)
            arm = f'conv{depth}_amp2_relative{target}_eta0_seed{seed}'
            config.update(study_id=STUDY, arm_id=arm, seed=seed, init_checkpoint_path=str(initializer))
            config['initialization'].update(seed=seed, checkpoint_sha256=accepted_identity)
            config['datasets']['mnist']['params'].update(shuffle_seed=seed, split_seed=0)
            config['reporting'].update(study_id=STUDY, arm_id=arm)
            reference = config['relative_output_calibration']
            config['relative_output_calibration'] = {
                key: copy.deepcopy(reference[key]) for key in
                ('target', 'T', 'K', 'definition', 'base_beta', 'injected_beta')}
            config['relative_output_calibration'].update(reference_seed=0,
                calibrated_for_this_seed=False, beta_frozen_across_seeds=True,
                actual_displacement_for_this_seed='unmeasured_may_differ', seed0_reference=reference)
            config['current_amp2_comparison'].update(parent_seed0_config=str(parent_path),
                parent_seed0_config_sha256=sha(ROOT/parent_path), model_seed=seed,
                shuffle_seed=seed, split_seed=0, reference_displacement_seed=0,
                fixed_seed0_beta=True, actual_displacement_for_this_seed='unmeasured_may_differ')
            path = CONFIGS/(arm+'.json')
            write_once(path, config)
            load_exact_config(ROOT/path)
            runtime_config = copy.deepcopy(config)
            runtime_config['init_checkpoint_path'] = str(ROOT/initializer)
            runtime = base._build_runtime(runtime_config, device=torch.device('cpu'),
                gradient_iterations=2*depth+2, nudging_mode='current', current_scale='auto')
            saved = torch.load(ROOT/initializer, map_location='cpu', weights_only=False)
            parameters = runtime['parameters']
            assert len(parameters) == len(saved['states'])
            for parameter, state in zip(parameters, saved['states']):
                assert parameter.state.dtype == state.dtype == torch.float32
                assert torch.equal(parameter.state, state)
                if str(parameter.name).startswith('Bias_'):
                    assert not bool(parameter.state.count_nonzero())
            shadow._convert_runtime_dtype(runtime, torch.float64)
            for parameter, state in zip(parameters, saved['states']):
                assert parameter.state.dtype == torch.float64
                assert torch.equal(parameter.state, state.to(torch.float64))
            state_hash = base._parameter_state_sha256(parameters)
            hashes.append(state_hash)
            provenance['initializer_checks'].append(dict(architecture=f'conv{depth}', seed=seed,
                accepted_checkpoint=str(accepted), checkpoint_sha256=accepted_identity,
                initial_parameter_state_sha256=state_hash, loaded_before_float64=True,
                accepted_tensors_identical=True, all_biases_zero=True, CPU_only=True))
            provenance['configs'].append(dict(path=str(path), sha256=sha(ROOT/path), arm_id=arm))
            del runtime, parameters, saved
        assert len(set(hashes)) == 3, f'Initial states not distinct for conv{depth}'
    provenance['distinct_initial_states_per_architecture'] = True
    write_once(CONFIGS/'provenance/preparation.json', provenance)
    print(json.dumps(dict(configs=6, initializer_checks=6, study_root=str(OUT),
                         independent_seeds=True, calibration='fixed seed0 reference; no new measurements')))


if __name__ == '__main__':
    main()
