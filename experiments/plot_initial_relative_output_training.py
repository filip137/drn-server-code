"""Offline validation and fixed-contrast plots for exp-007 (ordinary MNIST only).

Example:
  python -m experiments.plot_initial_relative_output_training \
    --study-root results/eqprop-conv123-relative-output-20260924-v1 \
    --production-root results/eqprop-conv123-relative-output-20260924-v1/production

No run is chosen by accuracy. Distinct duplicate attempts remain ambiguous unless
an operationally excluded attempt is explicitly named with --exclude-run.
Missing local bundles are labeled not_collected; this says nothing about remote
execution state. Explicitly confirmed user cancellations can be recorded with
--cancelled-arm, including arms that never created a local bundle.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from experiments.reporting import sha256_file, validate_run

ROOT = Path(__file__).resolve().parents[1]
TARGETS = (.8, 1.2, 1.6, 2.)
ARCHITECTURES = ('conv1', 'conv2', 'conv3')
SCHEMES = ('baseline', 'ours', 'legacy')


def read(path):
    return json.loads(Path(path).read_text())


def scientific_config(config):
    """Only initializer and dataset locations are transport-dependent."""
    config = copy.deepcopy(config)
    config['init_checkpoint_path'] = '<initializer-location>'
    config['datasets'][config['lab']['dataset_key']]['params']['root'] = '<dataset-location>'
    return config


def require(condition, message):
    if not condition:
        raise ValueError(message)


def arm_key(config):
    calibration = config['relative_output_calibration']
    return (f"conv{len(config['model_overrides'][config['lab']['model_key']]['conv_pipeline'])}",
            config['arm_id'].split('_')[1], float(calibration['target']))


def load_configs(config_root):
    configs = {}
    for path in sorted(Path(config_root).glob('*.json')):
        c = read(path)
        key = arm_key(c)
        require(key not in configs, f'Duplicate canonical config: {key}')
        arch, scheme, target = key
        require(arch in ARCHITECTURES and scheme in SCHEMES and target in TARGETS, f'Unexpected arm {key}')
        require(c['lab']['epochs'] == 10 and c['seed'] == 0, f'Wrong epoch/seed: {path}')
        require(c['evaluation']['official_test']['policy'] == 'disabled', 'Official test must be disabled')
        require(c['eqprop']['endpoint_read_noise_std'] == .0005, 'Wrong endpoint noise')
        scale = (c['model_base']['voltage_amp']/c['model_base']['current_amp'])**int(arch[-1])
        require(math.isclose(c['beta']*scale, c['eqprop']['injected_beta_B'], rel_tol=1e-13), 'Beta convention mismatch')
        require(c['relative_output_calibration']['measured']['relative_target_error'] <= .01, 'Unmatched calibration')
        configs[key] = (path, c)
    require(set(configs) == {(a,s,t) for a in ARCHITECTURES for s in SCHEMES for t in TARGETS},
            'Expected exactly the declared 36 canonical arms')
    return configs


def discover(production_roots, expected_arms, excluded=()):
    """Keep duplicate attempts visible; deduplicate only byte-identical copies."""
    attempts, seen, ignored_smokes = [], set(), 0
    excluded = {Path(p).resolve() for p in excluded}
    for root in production_roots:
        for path in sorted(Path(root).rglob('manifest.json')):
            manifest = read(path)
            if manifest.get('smoke'):
                ignored_smokes += 1
                continue
            arm = manifest.get('arm_id')
            if arm not in expected_arms:
                continue
            run = path.parent.resolve()
            if run in seen:
                continue
            seen.add(run)
            status = read(run/'status.json') if (run/'status.json').exists() else {'state':'missing_status'}
            identity = sha256_file(path)
            if status['state'] == 'complete':
                errors = validate_run(run)
                if errors:
                    status = dict(state='invalid',error=errors)
                else:
                    identity += ':'+sha256_file(run/'result.json')
            attempts.append(dict(arm_id=arm, run=str(run), state=status['state'],
                excluded=run in excluded, error=status.get('error'), manifest=manifest,
                identity=identity))
    grouped = {}
    for attempt in attempts:
        grouped.setdefault(attempt['arm_id'], []).append(attempt)
    return grouped, attempts, ignored_smokes


def epoch_history(run):
    records = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()]
    rows = [r for r in records if r.get('kind') == 'epoch']
    require([r['epoch'] for r in rows] == list(range(1, 11)), 'Expected exactly epochs 1 through 10')
    for row in rows:
        require(row.get('evaluation_split') == 'validation', 'Epoch is not validation')
        require(all(math.isfinite(row['metrics'][key]) for key in
                    ('validation_accuracy','validation_loss','train_accuracy','train_loss')), 'Nonfinite epoch')
        require(0 <= row['metrics']['validation_accuracy'] <= 1, 'Invalid accuracy range')
    return rows


def initializer_identity(config):
    """Reproduce mnist_train's ordered native-float64 initialization digest."""
    import torch
    path = Path(config['init_checkpoint_path'])
    if not path.is_absolute():
        path = ROOT/path
    require(sha256_file(path) == config['initialization']['checkpoint_sha256'], 'Initializer file hash mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    digest = hashlib.sha256(b'mnist-train-ordered-parameter-state/v1\0')
    names = [spec['name'].strip() for spec in checkpoint['schema']]
    require(names == config['parameter_order'], 'Initializer parameter order differs')
    for name, tensor in zip(names, checkpoint['states'], strict=True):
        tensor = tensor.to(torch.float64).contiguous()
        if name.startswith('Bias_'):
            require(bool((tensor == 0).all()), 'Nonzero initial bias')
        for part in (name.encode(), str(tensor.dtype).encode(), json.dumps(list(tensor.shape)).encode()):
            digest.update(part)
            digest.update(b'\0')
        digest.update(tensor.numpy().tobytes(order='C'))
    return digest.hexdigest()


def placement(manifest):
    runtime = manifest['runtime']
    target, host = str(runtime.get('target') or ''), str(runtime.get('host') or '')
    require(runtime.get('device') == 'cuda', 'Training did not record CUDA')
    require(bool(host), 'Missing actual execution host')
    if runtime.get('slurm_job_id') and (target.startswith('jz-') or target=='jean-zay-task6-recovery'):
        return host, 'declared_jean_zay_V100', 'Jean_Zay_V100'
    aliases = {'local':'RTX3090','main':'RTX3090','nom-cool-2':'RTX3090',
               'nom-cool-1':'RTX3090','akib':'RTX3080','akibscomputer':'RTX3080',
               'integnano-akib':'RTX3080','fifi':'RTX5090','loulou':'RTX5090',
               'trex':'RTX5090','riri':'RTX5090'}
    gpu = aliases.get(target) or aliases.get(host)
    require(gpu is not None, f'Unrecognized execution host/class: {host}/{target}')
    return host, 'declared_'+gpu, host


def validate_v100_wrappers(study_root, attempts):
    wrappers=[Path(study_root)/'transport/conv3.slurm']
    if any(a['manifest'].get('study_id')==Path(study_root).name and
           a['manifest']['runtime'].get('target')=='jean-zay-task6-recovery' for a in attempts):
        wrappers.append(Path(study_root)/'transport/conv3-task6-recovery.slurm')
    for wrapper in wrappers:
        require(re.search(r'^#SBATCH\s+--constraint[=\s]+v100(?:-16g|-32g)?\s*$',
                          wrapper.read_text(),re.MULTILINE) is not None,
                f'Missing declared V100 execution contract: {wrapper}')


def validate_complete(run, expected, expected_archive, initial_hash, validation_indices_sha256):
    import numpy as np
    import torch
    errors = validate_run(run)
    require(not errors, f'Invalid reporting bundle: {errors}')
    manifest, result, metrics, used, metadata = (read(run/name) for name in
        ('manifest.json','result.json','metrics.json','config.used.json','run_metadata.json'))
    require(not manifest.get('smoke') and not result.get('smoke'), 'Smoke cannot count as training')
    require(manifest['study_id'] == result['study_id'] == expected['study_id'], 'Unexpected study identity')
    require(result['completion']['criteria_met'], 'Completion criteria not met')
    require(manifest['configuration']['epochs'] == metadata['epochs'] == 10, 'Short epoch budget')
    require(scientific_config(used) == scientific_config(expected), 'Scientific config changed')
    require(scientific_config(manifest['configuration']['resolved']) == scientific_config(expected), 'Manifest config changed')
    require(manifest['git']['source_archive_sha256'] == expected_archive, 'Scientific source archive changed')
    require(metadata['beta'] == expected['beta'], 'Actual internal training beta changed')
    require(metadata['seed'] == metrics['seed'] == expected['seed'] == 0, 'Model seed changed')
    require('--skip-terminal-official-test' in manifest['command'], 'Missing explicit official-test skip flag')
    require(not any(flag in manifest['command'] for flag in ('--max-batches','--max-test-batches')), 'Shortened data budget')
    for owner in (manifest, result):
        require(owner['dataset']['official_test_read'] is False and
                owner['dataset']['official_test_read_during_training'] is False and
                owner['dataset']['evaluation_split'] == 'validation', 'Official-test or split violation')
    require(metrics['official_test_evaluations'] == metrics['official_test_examples'] == 0, 'Official-test access')
    require(metrics['training_algorithm'] == 'EP' and metrics['runtime_dtype'] == 'float64', 'Training algorithm/dtype mismatch')
    require(metrics['initial_parameter_state_sha256'] == initial_hash, 'Actual initialization mismatch')
    for key in ('endpoint_read_noise_std','endpoint_read_noise_seed','variant','nudging_mode','current_scale','normalize_current_scale','input_read_noise'):
        require(metrics['eqprop'][key] == expected['eqprop'][key], f'Actual EqProp setting changed: {key}')
    depth = int(arm_key(expected)[0][-1])
    require(metrics['eqprop_endpoint_read_noise_draw_count'] == 2*(depth+1)*3438*10, 'Incomplete endpoint noise acquisition coverage')
    require(metrics['final_learning_rate'] == expected['lr'], 'Learning rates changed')
    provenance = metrics['dataset_provenance']
    require(provenance['schema'] == 'mnist-train-validation-split/v1' and provenance['source_split'] == 'train', 'Wrong dataset provenance')
    require(provenance['split_seed'] == provenance['shuffle_seed'] == 0, 'Split/shuffle seed changed')
    require(provenance['validation_indices_sha256'] == validation_indices_sha256,
            'Training validation partition differs from frozen calibration cohort')
    require(len(provenance['train_batch_order_sha256']) == 10, 'Missing epoch minibatch-order identities')
    rows = epoch_history(run)
    for name, key in [('accuracy_test','validation_accuracy'),('loss_test','validation_loss'),
                      ('accuracy_train','train_accuracy'),('loss_train','train_loss')]:
        values = np.load(run/f'{name}.npy', allow_pickle=False)
        require(values.shape == (10,) and bool(np.isfinite(values).all()), f'Invalid {name} history')
        require(bool(np.allclose(values, [r['metrics'][key] for r in rows], rtol=1e-12, atol=1e-12)), f'{name} differs from epoch records')
    accuracy = rows[-1]['metrics']['validation_accuracy']
    validation_correct = round(accuracy*5000)
    require(math.isclose(accuracy,validation_correct/5000,rel_tol=0,abs_tol=1e-12),
            'Accuracy is not a 5,000-example validation fraction')
    require(math.isclose(accuracy,metrics['final_validation_accuracy'],abs_tol=1e-12), 'Terminal accuracy differs from epoch 10')
    require(math.isclose(accuracy,result['terminal_metrics']['validation']['final_accuracy'],abs_tol=1e-12), 'Result accuracy differs from epoch 10')
    for role in ('best','final'):
        checkpoint = torch.load(run/f'{role}_model.pt', map_location='cpu', weights_only=True)
        require([spec['name'].strip() for spec in checkpoint['schema']] == expected['parameter_order'],
                'Checkpoint parameter identity changed')
        for spec,tensor in zip(checkpoint['schema'],checkpoint['states'],strict=True):
            require(tensor.dtype == torch.float64 and bool(torch.isfinite(tensor).all()), 'Nonfinite/non-float64 checkpoint')
            name = spec['name'].strip()
            if name.startswith('Bias_'):
                require(bool((tensor == 0).all()), 'Frozen bias changed')
            else:
                require(bool(((tensor >= 0)&(tensor <= 100)).all()), 'Weight outside [0,100]')
    host, gpu_class, placement_group = placement(manifest)
    return dict(final_validation_percent=100*accuracy,
        validation_correct=validation_correct,validation_examples=5000,
        final_validation_loss=rows[-1]['metrics']['validation_loss'],
        best_validation_percent=100*max(r['metrics']['validation_accuracy'] for r in rows),
        host=host, gpu_class=gpu_class, placement_group=placement_group,
        initial_state_sha256=initial_hash, dataset_provenance=provenance,
        source_archive_sha256=expected_archive, epochs=rows,
        cuda_peak_allocated_bytes=metrics.get('cuda_peak_allocated_bytes'))


def contrasts(rows):
    """All twelve prespecified contrasts, with no best-target selection."""
    indexed = {(r['architecture'],r['scheme'],r['target']):r for r in rows}
    pairs, means = [], []
    for arch in ARCHITECTURES:
        for target in TARGETS:
            ours, legacy = (indexed[(arch,scheme,target)] for scheme in ('ours','legacy'))
            pair = dict(architecture=arch,target=target,state='incomplete',ours_minus_legacy_pp=None,
                        ours_minus_legacy_correct=None,
                        ours_state=ours['state'],legacy_state=legacy['state'],issue='')
            if ours['state'] == legacy['state'] == 'complete':
                keys = ('placement_group','gpu_class','initial_state_sha256','dataset_provenance','source_archive_sha256')
                mismatches = [key for key in keys if ours[key] != legacy[key]]
                if mismatches:
                    pair.update(state='mismatched',issue=','.join(mismatches))
                else:
                    difference = ours['validation_correct']-legacy['validation_correct']
                    pair.update(state='complete',ours_minus_legacy_pp=difference/50,
                                ours_minus_legacy_correct=difference)
            pairs.append(pair)
        valid = [p for p in pairs if p['architecture']==arch and p['state']=='complete']
        average = sum(p['ours_minus_legacy_correct'] for p in valid)/200 if len(valid)==4 else None
        means.append(dict(architecture=arch,complete_pairs=len(valid),expected_pairs=4,
            four_target_mean_pp=average,
            state='complete' if len(valid)==4 else 'partial',
            included_targets=[p['target'] for p in valid]))
    return pairs, means


def collect(configs, grouped, expected_archive, validation_indices_sha256=None):
    rows, initial_hashes = [], {}
    for case_key,(path,c) in sorted(configs.items()):
        arch,scheme,target = case_key[:3]
        row = dict(architecture=arch,scheme=scheme,target=target,arm_id=c['arm_id'],state='not_collected',
            injected_beta=c['eqprop']['injected_beta_B'],base_beta=c['beta'],
            measured_initial_ratio=c['relative_output_calibration']['measured']['relative_output_rms'],
            config=str(path),runs=[],issue='',final_validation_percent=None)
        if len(case_key) == 4:
            row['sigma'] = case_key[3]
            row['source_study_id'] = c['study_id']
        candidates = [a for a in grouped.get(c['arm_id'],[]) if not a['excluded']]
        row['runs'] = [a['run'] for a in grouped.get(c['arm_id'],[])]
        # Identical manifest copies represent one attempt; distinct attempts are
        # never ranked using completion or accuracy.
        unique = {a['identity']:a for a in candidates}
        if len(unique)>1:
            row.update(state='ambiguous',issue='Multiple distinct attempts; explicitly exclude operationally invalid attempts')
        elif unique:
            attempt = next(iter(unique.values()))
            row['state'] = attempt['state']
            if attempt['state'] == 'complete':
                try:
                    key = c['initialization']['checkpoint_sha256']
                    if key not in initial_hashes:
                        initial_hashes[key] = initializer_identity(c)
                    archive = expected_archive[c['study_id']] if isinstance(expected_archive,dict) else expected_archive
                    row.update(validate_complete(Path(attempt['run']),c,archive,initial_hashes[key],
                                                 validation_indices_sha256), state='complete')
                except (ValueError,KeyError,OSError,TypeError,RuntimeError,IndexError) as error:
                    row.update(state='invalid',issue=str(error))
            else:
                row['issue'] = json.dumps(attempt['error']) if attempt.get('error') else ''
        rows.append(row)
    # Full cohort/order identity is common across this entire selection study.
    identities = {json.dumps(r['dataset_provenance'],sort_keys=True) for r in rows if r['state']=='complete'}
    if len(identities)>1:
        for row in rows:
            if row['state']=='complete':
                row.update(state='invalid',issue='Cross-run dataset/split/minibatch-order identity mismatch')
    groups = {(r['architecture'],r['target'],r.get('sigma')) for r in rows}
    for group in groups:
        triplet = [r for r in rows if (r['architecture'],r['target'],r.get('sigma')) == group
                   and r['state'] == 'complete']
        if len({(r['placement_group'],r['gpu_class']) for r in triplet}) > 1:
            for row in triplet:
                row.update(state='invalid',issue='Within-target three-scheme placement mismatch')
    return rows


def apply_cancellations(rows, cancelled_arms):
    """Annotate confirmed user decisions without discarding completed evidence."""
    cancelled_arms = set(cancelled_arms)
    require(cancelled_arms <= {row['arm_id'] for row in rows}, 'Unknown cancelled arm')
    for row in rows:
        if row['arm_id'] not in cancelled_arms:
            continue
        row['user_cancellation_recorded'] = True
        row['observed_bundle_state'] = row['state']
        # Invalid or ambiguous evidence stays explicitly invalid/ambiguous.
        if row['state'] not in ('complete', 'invalid', 'ambiguous'):
            row['state'] = 'cancelled_by_user'
            row['issue'] = ('User-directed cancellation; no automatic replacement. '+row.get('issue','')).strip()
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row if key not in ('epochs','manifest')))
    with path.open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value
                             for key,value in row.items() if key in fields})


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors = {'baseline':'#666666','ours':'#0072B2','legacy':'#D55E00'}
    lookup = {(r['architecture'],r['scheme'],r['target']):r for r in rows}
    fig, axes = plt.subplots(1,3,figsize=(12,3.6),sharey=True)
    for ax,arch in zip(axes,ARCHITECTURES):
        for scheme in SCHEMES:
            values = [lookup[(arch,scheme,t)]['final_validation_percent'] if lookup[(arch,scheme,t)]['state']=='complete' else np.nan for t in TARGETS]
            ax.plot(TARGETS,values,'o-',label=scheme,color=colors[scheme])
        ax.set(title=arch.upper(),xlabel='Initial output D/F',xticks=TARGETS)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Epoch-10 validation accuracy (%)')
    axes[-1].legend()
    fig.suptitle('Seed 0, endpoint noise σ=5e−4; missing/invalid arms remain gaps')
    fig.tight_layout()
    fig.savefig(output/'epoch10_accuracy_vs_target.png',dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(3,4,figsize=(14,9),sharex=True,sharey=True)
    for i,arch in enumerate(ARCHITECTURES):
        for j,target in enumerate(TARGETS):
            ax=axes[i,j]
            for scheme in SCHEMES:
                row=lookup[(arch,scheme,target)]
                if row['state']=='complete':
                    ax.plot(range(1,11),[100*r['metrics']['validation_accuracy'] for r in row['epochs']],label=scheme,color=colors[scheme])
            missing = [s for s in SCHEMES if lookup[(arch,s,target)]['state']!='complete']
            ax.set(title=f'{arch.upper()}, D/F={target:g}',xticks=(1,5,10))
            if missing:
                ax.text(.02,.04,'Not validated: '+', '.join(missing),transform=ax.transAxes,fontsize=7)
            ax.grid(alpha=.2)
            if i==2: ax.set_xlabel('Epoch')
            if j==0: ax.set_ylabel('Validation accuracy (%)')
    handles=[plt.Line2D([],[],color=colors[s],label=s) for s in SCHEMES]
    fig.legend(handles=handles,loc='upper center',ncol=3)
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(output/'validation_epoch_curves.png',dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,required=True)
    parser.add_argument('--production-root',type=Path,action='append',required=True)
    parser.add_argument('--config-root',type=Path,default=ROOT/'configs/conv/eqprop_conv123_relative_output_20260924')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--calibration-run',type=Path,
        help='Validated full calibration bundle; defaults to the collected Loulou bundle under the study')
    parser.add_argument('--exclude-run',type=Path,action='append',default=[])
    parser.add_argument('--cancelled-arm',action='append',default=[],
        help='Canonical arm_id explicitly confirmed cancelled by the user; repeat per arm')
    args=parser.parse_args()
    configs=load_configs(args.config_root)
    calibration_run = args.calibration_run or (args.study_root/'calibration_collected_loulou'/'results'/
        'eqprop-conv123-relative-output-20260924-v1'/'calibration'/'all')
    require(not validate_run(calibration_run), 'Invalid or missing full calibration bundle')
    calibration_result=read(calibration_run/'result.json')
    calibration_metrics=calibration_result['terminal_metrics']
    require(not calibration_result['smoke'] and calibration_metrics['matches']==36 and
            calibration_metrics['official_test_read'] is False and
            calibration_metrics['optimizer_steps_applied'] is False, 'Incomplete or invalid calibration')
    calibration_cohort=read(calibration_run/'cohort.json')
    expected_archive=(args.study_root/'transport/source-archive-sha256.txt').read_text().strip()
    grouped,attempts,ignored=discover(args.production_root,{c['arm_id'] for _,c in configs.values()},args.exclude_run)
    validate_v100_wrappers(args.study_root,attempts)
    rows=collect(configs,grouped,expected_archive,calibration_cohort['validation_indices_sha256'])
    apply_cancellations(rows,args.cancelled_arm)
    pairs,means=contrasts(rows)
    output=args.output_dir or args.study_root/'analysis'
    output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'training_coverage_36.csv',rows)
    write_csv(output/'ours_minus_legacy_12.csv',pairs)
    write_csv(output/'architecture_means.csv',means)
    write_csv(output/'attempts.csv',attempts or [dict(arm_id='',run='',state='no_collected_attempts')])
    summary=dict(expected_arms=36,complete_arms=sum(r['state']=='complete' for r in rows),
        coverage={state:sum(r['state']==state for r in rows) for state in sorted({r['state'] for r in rows})},
        ignored_smoke_bundles=ignored,excluded_runs=[str(p.resolve()) for p in args.exclude_run],
        user_cancelled_arms=args.cancelled_arm,
        complete_pairs=sum(p['state']=='complete' for p in pairs),
        source_archive_sha256=expected_archive,ordinary_mnist_validation_only=True,
        calibration_cohort_sha256=calibration_cohort['cohort_sha256'],
        host_classes='Declared inventory/Slurm contract; actual host equality required for workstation pairs',
        primary_summary='Four-target architecture means only when all four fixed pairs are complete; no partial mean substitutes for missing targets')
    (output/'validation_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    plot(rows,output)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
