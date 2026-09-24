"""Prepare the requested baseline beta 100/200/300 read-only audit."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'eqprop-beta-selection-audit-20260914-v1'
OUT = ROOT / 'results' / STUDY
CONFIGS = ROOT / 'configs/conv/eqprop_beta_selection_audit_20260914_v1/v2'
PARENT = ROOT / 'results/paper-training-completion-20260911-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if path.exists() and path.read_text() != content:
        raise ValueError(f'Refusing to replace frozen input: {path}')
    path.write_text(content)


def main():
    template_path = PARENT / 'checks/configs/wide_conv1_legacy.json'
    sys.argv.extend(['--config', str(template_path)])
    path = ROOT / 'experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py'
    spec = importlib.util.spec_from_file_location('_prepare_beta_gate', path)
    gate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = gate
    spec.loader.exec_module(gate)
    base = gate.base
    template = json.loads(template_path.read_text())
    template['source_contract']['runtime_source_root'] = str(PARENT / 'source')
    for name, expected in template['source_contract']['runtime_code_sha256'].items():
        if sha(PARENT / 'source' / name) != expected:
            raise ValueError(f'The preserved scientific runtime changed: {name}')
    original = json.loads((ROOT / 'paper_ready_results/bundles/table1_wide_bptt/conv1/baseline/seed0/config.used.json').read_text())
    signature = base._dataset_signature(original, data_root=Path(template['dataset']['root']), batch_size=16)
    params = dict(signature['params'], device=base.torch.device('cpu'))
    loader = base._resolve_callable(signature['factory'])(**params).build()
    validation = [int(i) for i in loader.validation_indices]
    historical = validation[:64]
    rng = base.np.random.Generator(base.np.random.PCG64(2026091401))
    permuted = [int(i) for i in rng.permutation(validation[64:])]
    selection, confirmation = permuted[:512], permuted[512:1024]
    assert len(set(historical + selection + confirmation)) == 1088
    datasets = {}
    for phase, indices, excluded in [('selection', historical + selection, confirmation),
                                      ('confirmation', historical + confirmation, selection)]:
        batches, cohort = base._build_validation_cohort(
            original, data_root=Path(template['dataset']['root']), batch_size=16,
            example_count=len(indices), source_indices=indices, excluded_source_indices=excluded)
        guards = [{k: b[k] for k in ('batch_index', 'source_indices_sha256', 'payload_sha256')} for b in batches]
        if guards[:4] != template['dataset']['expected_batch_guards']:
            raise ValueError('Historical four batches did not reproduce exactly.')
        datasets[phase] = dict(template['dataset'], source_indices=indices,
                              excluded_source_indices=excluded, example_count=len(indices), batch_count=len(batches),
                              expected_batch_guards=guards, cohort_role=phase,
                              historical_batch_count=4, new_batch_count=32)
        write(OUT / 'cohorts' / f'{phase}.json', cohort)
    write(OUT / 'cohorts/partition.json', dict(seed=2026091401, algorithm='NumPy PCG64',
          historical=historical, selection=selection, confirmation=confirmation,
          validation_indices_sha256=loader.validation_indices_hash,
          official_test_read=False, new_cohorts_disjoint=True))
    files = ['analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py',
             'audit_eqprop_float64_shadow.py', 'analyze_conv_eqprop_bptt_beta_tk_displacement.py',
             'analyze_conv_eqprop_bptt_checkpoint_gradients.py']
    snapshot = {}
    for name in files:
        source = ROOT / 'experiments' / name
        dest = OUT / 'source/experiments' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha(source) != sha(dest):
            raise ValueError('Frozen analyzer already exists with a different hash.')
        shutil.copy2(source, dest)
        snapshot[f'experiments/{name}'] = sha(dest)
    write(OUT / 'source_identity.json', snapshot)
    prepared = []
    for architecture in ('conv3', 'conv2', 'conv1'):
        depth = int(architecture[-1]); tk = {1:4, 2:6, 3:8}[depth]
        for phase in datasets:
            for seed in ([0] if phase == 'selection' else [0, 1, 2]):
                source = ROOT / f'paper_ready_results/bundles/table1_wide_bptt/{architecture}/baseline/seed{seed}'
                manifest = json.loads((source / 'manifest.json').read_text())
                asset = PARENT / f'assets/wide_kaiming/{architecture}/seed{seed}.pt'
                for beta in [100, 200, 300]:
                    config = copy.deepcopy(template)
                    config.update(study_id=STUDY, scientific_question='Does baseline beta 100/200/300 meet the frozen gradient gates on broader selection and disjoint confirmation batches?')
                    config['source_contract'].update(source_study_root=str(source.parent), allow_partial_schemes=True)
                    config['dataset'] = datasets[phase]
                    config['cases'] = [dict(architecture=architecture, scheme='baseline', model_seed=seed,
                        run_id=source.name, run_dir=str(source), T=tk, K=tk, output_row_exponent_L=depth,
                        base_beta=beta, injected_beta=beta,
                        production_source_commit=manifest['git']['commit'],
                        production_source_archive_sha256=manifest['git'].get('source_archive_sha256'),
                        source_file_sha256={n:sha(source/n) for n in config['source_contract']['required_source_files']},
                        initializer_checkpoint_path=str(asset), initializer_checkpoint_sha256=sha(asset))]
                    config['completion'].update(production_replay_count=72, production_layer_comparison_count=72*(depth+1))
                    name = f'{phase}_{architecture}_baseline_seed{seed}_beta{beta}'
                    dest = CONFIGS / f'{name}.json'
                    write(dest, config)
                    prepared.append(dict(name=name, phase=phase, architecture=architecture, seed=seed, beta=beta,
                                         path=str(dest.relative_to(ROOT)), sha256=sha(dest)))
    write(OUT / 'prepared_configs_v2.json', prepared)
    print(json.dumps(dict(study=STUDY, configs=len(prepared), selection_cases=9,
                         confirmation_cases='only the frozen selected beta, at most 9',
                         source_files=snapshot, historical_guards_reproduced=True,
                         new_cohorts_disjoint=True), indent=2))


if __name__ == '__main__':
    main()
