"""Validate exp008's 135 active cells, including 18 explicitly reused exp007 cells.

Pass --production-root repeatedly for all authoritative local production roots from
both studies. Missing bundles remain not_collected. Distinct attempts are never
ranked by accuracy; --exclude-run names an operational exclusion explicitly.
The 18 inherited cells outside the narrowed exp008 domain remain in a separate
coverage table, including the three user-cancelled Conv2 target1.6 cells. They do
not enter exp008 contrasts or replace missing active cells.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.plot_initial_relative_output_training import (
    ARCHITECTURES, ROOT, SCHEMES, apply_cancellations, arm_key, collect, discover,
    load_configs, read, require, validate_v100_wrappers, write_csv,
)
from experiments.reporting import sha256_file, validate_run

NOISES = (.0005, .001, .002)
TARGETS_BY_ARCH = {'conv1': (.8, 1.2, 1.6, 2., 2.5, 3., 6.),
                   'conv2': (2., 2.5, 3., 6.), 'conv3': (2., 2.5, 3., 6.)}
OLD_ID = 'eqprop-conv123-relative-output-20260924-v1'
NEW_ID = 'eqprop-conv123-relative-output-noise-extension-20260924-v1'
CANCELLED = tuple(f'conv2_{scheme}_relative1p6_sigma5em4_seed0' for scheme in SCHEMES)


def active_keys():
    return {(a,s,t,n) for a in ARCHITECTURES for s in SCHEMES
            for t in TARGETS_BY_ARCH[a] for n in NOISES}


def combined_configs(extension_root, prior_root):
    prior = {(*key,.0005):value for key,value in load_configs(prior_root).items()}
    extension = {}
    for path in sorted(Path(extension_root).glob('*.json')):
        config = read(path)
        key = (*arm_key(config),config['eqprop']['endpoint_read_noise_std'])
        require(key not in extension and key not in prior, f'Duplicate/repeated training cell {key}')
        require(key in active_keys(), f'Out-of-scope extension config {key}')
        require(config['study_id'] == NEW_ID, 'Wrong extension study identity')
        require(config['lab']['epochs'] == 10 and config['seed'] == 0, 'Wrong epoch/seed')
        require(config['evaluation']['official_test']['policy'] == 'disabled', 'Official test enabled')
        scale = (config['model_base']['voltage_amp']/config['model_base']['current_amp'])**int(key[0][-1])
        require(math.isclose(config['beta']*scale,config['eqprop']['injected_beta_B'],rel_tol=1e-13),
                'Internal/injected beta mismatch')
        extension[key] = (path,config)
    require(set(extension) == active_keys()-set(prior) and len(extension)==117,
            'Expected exactly 117 new configs, without repeating inherited cells')
    require(all(c['study_id']==OLD_ID for _,c in prior.values()), 'Wrong inherited study identity')
    return prior | extension


def calibration_evidence(run, expected_matches):
    require(not validate_run(run), f'Invalid calibration bundle: {run}')
    result = read(run/'result.json')
    metrics = result['terminal_metrics']
    require(not result['smoke'] and metrics['matches']==expected_matches and
            metrics['official_test_read'] is False and metrics['optimizer_steps_applied'] is False and
            metrics['all_parameters_unchanged'] is True and metrics['all_bias_tensors_exact_zero'] is True,
            'Incomplete or invalid clean calibration')
    matches = read(run/'matches.json')
    require(len(matches)==expected_matches, 'Calibration count mismatch')
    return read(run/'cohort.json'), matches


def validate_calibration_reuse(configs, runs):
    measured, cohort = {}, None
    for run, count in runs:
        actual, matches = calibration_evidence(run,count)
        if cohort is None:
            cohort = actual
        require(actual['cohort_sha256']==cohort['cohort_sha256'] and
                actual['validation_indices_sha256']==cohort['validation_indices_sha256'],
                'Calibration cohort changed between targets')
        for match in matches:
            key = (match['architecture'],match['scheme'],match['target'])
            require(key not in measured, 'Repeated calibration context')
            measured[key] = match
    for key,(path,config) in configs.items():
        match = measured[key[:3]]
        calibration = config['relative_output_calibration']
        require(config['eqprop']['injected_beta_B']==match['injected_beta'] and
                config['beta']==match['base_beta'] and calibration['measured']==match['history'][-1],
                f'Config differs from measured calibration: {path}')
        require(calibration['measured']['relative_target_error']<=.01, 'Calibration target missed')
        if key[3] != .0005:
            reused = ROOT/calibration['reused_from_training_config']
            require(sha256_file(reused)==calibration['reused_from_training_config_sha256'],
                    'Reused calibration source config hash changed')
            original = read(reused)
            require(arm_key(original)==key[:3] and original['eqprop']['endpoint_read_noise_std']==.0005 and
                    original['beta']==config['beta'] and original['relative_output_calibration']['measured']==calibration['measured'],
                    'Higher-noise config does not reuse the measured clean beta')
    return cohort


def contrasts(rows):
    indexed = {(r['architecture'],r['scheme'],r['target'],r['sigma']):r for r in rows}
    require(set(indexed)==active_keys(), 'Contrast input must contain all 135 active cells')
    pairs, means = [], []
    for arch in ARCHITECTURES:
        targets = TARGETS_BY_ARCH[arch]
        for sigma in NOISES:
            group = []
            for target in targets:
                ours, legacy = (indexed[(arch,s,target,sigma)] for s in ('ours','legacy'))
                pair = dict(architecture=arch,sigma=sigma,target=target,state='incomplete',
                            ours_state=ours['state'],legacy_state=legacy['state'],
                            ours_minus_legacy_pp=None,ours_minus_legacy_correct=None,issue='')
                if ours['state']==legacy['state']=='complete':
                    keys=('placement_group','gpu_class','initial_state_sha256','dataset_provenance','source_archive_sha256')
                    mismatches=[key for key in keys if ours[key]!=legacy[key]]
                    if mismatches:
                        pair.update(state='mismatched',issue=','.join(mismatches))
                    else:
                        delta=ours['validation_correct']-legacy['validation_correct']
                        pair.update(state='complete',ours_minus_legacy_correct=delta,ours_minus_legacy_pp=delta/50)
                pairs.append(pair)
                group.append(pair)
            valid=[p for p in group if p['state']=='complete']
            average=sum(p['ours_minus_legacy_correct'] for p in valid)/(50*len(targets)) if len(valid)==len(targets) else None
            means.append(dict(architecture=arch,sigma=sigma,expected_pairs=len(targets),complete_pairs=len(valid),
                fixed_target_mean_pp=average,state='complete' if average is not None else 'partial',
                positive_pairs=sum(p['ours_minus_legacy_correct']>0 for p in valid),
                negative_pairs=sum(p['ours_minus_legacy_correct']<0 for p in valid),
                tied_pairs=sum(p['ours_minus_legacy_correct']==0 for p in valid),
                included_targets=[p['target'] for p in valid]))
    return pairs,means


def plot(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'baseline':'#666666','ours':'#0072B2','legacy':'#D55E00'}
    lookup={(r['architecture'],r['scheme'],r['target'],r['sigma']):r for r in rows}
    fig,axes=plt.subplots(3,3,figsize=(13,10),sharey=True)
    for i,arch in enumerate(ARCHITECTURES):
        for j,sigma in enumerate(NOISES):
            ax=axes[i,j];targets=TARGETS_BY_ARCH[arch]
            for scheme in SCHEMES:
                selected=[lookup[(arch,scheme,t,sigma)] for t in targets]
                ax.plot(targets,[r['final_validation_percent'] if r['state']=='complete' else np.nan for r in selected],
                        'o-',label=scheme,color=colors[scheme])
            ax.set(title=f'{arch.upper()}, sigma={sigma:g}',xlabel='Initial output D/F',xticks=targets)
            ax.grid(alpha=.2)
            if j==0:ax.set_ylabel('Epoch-10 validation accuracy (%)')
    axes[0,-1].legend()
    fig.suptitle('Seed 0; missing/invalid arms remain gaps; 18 original cells reused')
    fig.tight_layout()
    fig.savefig(output/'epoch10_accuracy_vs_target_and_noise.png',dpi=180)
    plt.close(fig)
    for arch in ARCHITECTURES:
        targets=TARGETS_BY_ARCH[arch]
        fig,axes=plt.subplots(3,len(targets),figsize=(3*len(targets),9),sharex=True,sharey=True)
        for i,sigma in enumerate(NOISES):
            for j,target in enumerate(targets):
                ax=axes[i,j];missing=[]
                for scheme in SCHEMES:
                    row=lookup[(arch,scheme,target,sigma)]
                    if row['state']=='complete':
                        ax.plot(range(1,11),[100*r['metrics']['validation_accuracy'] for r in row['epochs']],
                                color=colors[scheme],label=scheme)
                    else:missing.append(scheme)
                ax.set(title=f'D/F={target:g}, sigma={sigma:g}',xticks=(1,5,10))
                if missing:ax.text(.02,.04,'Not validated: '+', '.join(missing),transform=ax.transAxes,fontsize=6)
                ax.grid(alpha=.2)
                if i==2:ax.set_xlabel('Epoch')
                if j==0:ax.set_ylabel('Validation accuracy (%)')
        fig.suptitle(arch.upper()+' — seed 0')
        fig.legend(handles=[plt.Line2D([],[],color=colors[s],label=s) for s in SCHEMES],loc='upper right',ncol=3)
        fig.tight_layout(rect=(0,0,1,.96))
        fig.savefig(output/f'{arch}_validation_epoch_curves.png',dpi=180)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root',type=Path,required=True)
    parser.add_argument('--prior-study-root',type=Path,required=True)
    parser.add_argument('--production-root',type=Path,action='append',required=True)
    parser.add_argument('--config-root',type=Path,default=ROOT/'configs/conv/eqprop_conv123_relative_output_noise_extension_20260924')
    parser.add_argument('--prior-config-root',type=Path,default=ROOT/'configs/conv/eqprop_conv123_relative_output_20260924')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--exclude-run',type=Path,action='append',default=[])
    args=parser.parse_args()
    configs=combined_configs(args.config_root,args.prior_config_root)
    cohort=validate_calibration_reuse(configs,[(args.prior_study_root/'calibration_collected_loulou/results'/OLD_ID/'calibration/all',36),
        (args.study_root/'calibration/all',18),(args.study_root/'target6/calibration/all',9)])
    archives={OLD_ID:(args.prior_study_root/'transport/source-archive-sha256.txt').read_text().strip(),
              NEW_ID:(args.study_root/'transport/source-archive-sha256.txt').read_text().strip()}
    grouped,attempts,ignored=discover(args.production_root,{c['arm_id'] for _,c in configs.values()},args.exclude_run)
    for study in (args.prior_study_root,args.study_root):
        validate_v100_wrappers(study,attempts)
    active=collect({k:v for k,v in configs.items() if k in active_keys()},grouped,archives,
                   cohort['validation_indices_sha256'])
    outside=collect({k:v for k,v in configs.items() if k not in active_keys()},grouped,archives,
                    cohort['validation_indices_sha256'])
    apply_cancellations(outside,CANCELLED)
    for row in active+outside:
        row['expected_source_archive_sha256']=archives[row['source_study_id']]
    require(len(active)==135 and len(outside)==18, 'Combined study coverage mismatch')
    pairs,means=contrasts(active)
    output=args.output_dir or args.study_root/'analysis';output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'training_coverage_135.csv',active)
    write_csv(output/'inherited_out_of_scope_18.csv',outside)
    write_csv(output/'ours_minus_legacy_45.csv',pairs)
    write_csv(output/'architecture_noise_means_9.csv',means)
    write_csv(output/'attempts.csv',attempts or [dict(state='no_collected_attempts')])
    placements=[dict(architecture=r['architecture'],sigma=r['sigma'],target=r['target'],scheme=r['scheme'],
                     host=r.get('host'),gpu_class=r.get('gpu_class'),source_study_id=r['source_study_id'])
                for r in active if r['state']=='complete']
    write_csv(output/'completed_placements.csv',placements or [dict(state='no_validated_completions')])
    changed_placements=[]
    for arch in ARCHITECTURES:
        for target in TARGETS_BY_ARCH[arch]:
            selected=[r for r in active if r['architecture']==arch and r['target']==target and r['state']=='complete']
            if len({(r['placement_group'],r['gpu_class']) for r in selected})>1:
                changed_placements.append(dict(architecture=arch,target=target,
                    placements=[dict(sigma=r['sigma'],scheme=r['scheme'],host=r['host'],gpu_class=r['gpu_class']) for r in selected]))
    summary=dict(expected_active_cells=135,new_cells=117,reused_cells=18,
        inherited_out_of_scope_cells=18,user_cancelled_out_of_scope_arms=list(CANCELLED),
        complete_active_cells=sum(r['state']=='complete' for r in active),
        coverage={s:sum(r['state']==s for r in active) for s in sorted({r['state'] for r in active})},
        complete_pairs=sum(p['state']=='complete' for p in pairs),expected_pairs=45,
        ignored_smoke_bundles=ignored,excluded_runs=[str(p.resolve()) for p in args.exclude_run],
        source_archive_sha256_by_study=archives,calibration_cohort_sha256=cohort['cohort_sha256'],
        observed_cross_noise_placement_changes=changed_placements,
        ordinary_mnist_validation_only=True,
        primary_summary='Seven fixed targets for Conv1; four for Conv2/3; each architecture/noise mean requires every pair. No partial mean substitution.',
        limitations='One seed; inherited cells selected retrospectively. Initial D/F is not absolute signal or SNR. Cross-noise host changes are visible in completed_placements.csv.')
    (output/'validation_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    plot(active,output)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
