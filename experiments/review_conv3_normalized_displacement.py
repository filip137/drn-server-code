"""Join existing Conv3 endpoint statistics to their parent training outcomes.

CPU-only evidence review: no model construction, checkpoint replay or training.
Historical endpoints lack absolute nudged RMS, so their D/P stays unavailable.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from experiments.reporting import atomic_write_json, sha256_file, validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'results/conv3-clean-eqprop-rms-beta-20260924-v1'
COLORS = {'baseline': '#4c566a', 'ours': '#0072b2', 'legacy': '#d55e00'}
PARAMETERS = ['ConvWeight_0', 'ConvWeight_1', 'ConvWeight_2', 'DenseWeight_0']
LAYERS = ['Conv1', 'Conv2', 'Conv3', 'Readout']


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pooled(rows, key):
    counts = [int(r['element_count']) for r in rows]
    values = [float(r[key]) for r in rows]
    assert all(n > 0 for n in counts) and all(math.isfinite(v) and v >= 0 for v in values)
    return math.sqrt(math.fsum(n*v*v for n, v in zip(counts, values))/sum(counts))


def main():
    out = STUDY/'normalization_review'
    training, phases, validated, sources = [], [], [], {}
    cohort_ids = set()

    def source(path):
        sources[str(path.relative_to(ROOT))] = sha256_file(path)

    def validate(path):
        errors = validate_run(path)
        assert not errors, (path, errors)
        validated.append(str(path.relative_to(ROOT)))
        source(path/'result.json')

    def config(name):
        p = ROOT/'configs/conv'/name
        source(p)
        return json.loads(p.read_text())

    def add_training(case, family):
        run = ROOT/case['bundle']
        validate(run)
        for name, expected in case['source_hashes'].items():
            assert sha256_file(run/name) == expected, (run, name)
        result = json.loads((run/'result.json').read_text())
        cfg = json.loads((run/'config.used.json').read_text())
        assert not result['dataset']['official_test_read']
        assert cfg['eqprop']['injected_beta_B'] == case['beta']
        assert cfg['eqprop']['endpoint_read_noise_std'] == case['sigma']
        metrics = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
        epochs = [r for r in metrics if r.get('kind') == 'epoch']
        assert [r['epoch'] for r in epochs] == list(range(1, 31))
        v = result['terminal_metrics']['validation']
        row = dict(case=case['name'], family=family, scheme=case['scheme'], training_gpu=case['gpu'],
                   training_sigma=case['sigma'], beta=case['beta'], checkpoint_epoch=30,
                   final_validation_pct=100*v['final_accuracy'], best_validation_pct=100*v['best_accuracy'],
                   final_validation_loss=v['final_loss'], epoch10_validation_pct=100*epochs[9]['metrics']['validation_accuracy'],
                   parent_run=case['bundle'], final_checkpoint_sha256=case['source_hashes']['final_model.pt'],
                   learning_rates=json.dumps(cfg['learning_rates_by_parameter'], sort_keys=True))
        training.append(row)
        return row

    def add_replay(parent, run):
        validate(run)
        manifest = json.loads((run/'manifest.json').read_text())
        cohort_ids.add(manifest['cohort_sha256'])
        rows = read_csv(run/'state_signals.csv')
        gradients = read_csv(run/'layer_metrics.csv')
        residuals = read_csv(run/'equilibrium_residuals.csv')
        assert len(rows) == 36*4
        for name in ['state_signals.csv', 'layer_metrics.csv', 'equilibrium_residuals.csv']:
            source(run/name)
        assert {(int(r['batch_index']), int(r['layer_index'])) for r in rows} == {
            (batch, layer) for batch in range(36) for layer in range(4)}
        for r in gradients:
            assert r['checkpoint_sha256'] == parent['final_checkpoint_sha256']
            assert int(r['T']) == int(r['K']) == 8
            assert float(r['injected_beta']) == parent['beta'] and r['scheme'] == parent['scheme']
        for layer, parameter in enumerate(PARAMETERS):
            sr = [r for r in rows if int(r['layer_index']) == layer]
            clean = [r for r in gradients if r['parameter_name'] == parameter and float(r['endpoint_read_noise_std']) == 0]
            noisy = [r for r in gradients if r['parameter_name'] == parameter and float(r['endpoint_read_noise_std']) == parent['training_sigma']]
            assert len(clean) == 36 and len(noisy) == 144
            rr = [r for r in residuals if int(r['layer_index']) == layer]
            assert rr
            f, d = pooled(sr, 'free_rms'), pooled(sr, 'centered_half_difference_rms')
            ratios = [float(r['centered_half_difference_rms'])/float(r['free_rms']) for r in sr]
            q = np.quantile(ratios, [.1, .5, .9])
            phases.append(dict(parent, layer=LAYERS[layer], layer_index=layer, free_rms=f, centered_rms=d,
                               normalized_D_over_F=d/f, normalized_D_over_P=None,
                               batch_ratio_p10=q[0], batch_ratio_median=q[1], batch_ratio_p90=q[2],
                               clean_cosine_median=float(np.median([float(r['cosine']) for r in clean])),
                               noisy_cosine_median=float(np.median([float(r['cosine']) for r in noisy])),
                               residual_failed_rows=sum(r['gate_passed'] != 'True' for r in rr),
                               T=8, K=8, replay_target=manifest['runtime']['target'],
                               replay_run=str(run.relative_to(ROOT))))

    p90 = config('eqprop_conv3_p90_final_noise_cosine_20260921.json')
    for case in p90['cases']:
        if case['gpu'] != 'A100' or case['sigma'] not in [0, .0003, .0005]:
            continue
        parent = add_training(case, 'p90')
        if case['sigma']:
            add_replay(parent, ROOT/'results/eqprop-conv3-trained-beta-noise-20260922-v1/collected/runs'/
                       (case['name']+'_factor_1'))
    p99 = config('eqprop_conv3_p99_trained_beta_replay_20260923.json')
    for case in p99['cases']:
        if case['checkpoint_epoch'] != 30:
            continue
        parent = add_training(case, 'p99')
        name = case['name']+'_T8K8_beta_'+format(case['beta'], '.12g').replace('.', 'p')
        add_replay(parent, ROOT/p99['output_root']/'collected/runs'/name)
    assert len(training) == 11 and len(phases) == 32 and len(cohort_ids) == 1

    calibration_path = ROOT/'results/eqprop-conv3-output-rms1-init-20260922-v1/analysis/layer_summary.csv'
    source(calibration_path)
    calibration = []
    for r in read_csv(calibration_path):
        calibration.append(dict(r, normalized_D_over_F=float(r['centered_half_difference_rms'])/float(r['free_rms']),
                                checkpoint_epoch=0, parent_training_outcome='not trained at these beta choices'))
    assert len(calibration) == 12
    for run in sorted((ROOT/'results/eqprop-conv3-output-rms1-init-20260922-v1/collected/workspace/output-retry/runs').iterdir()):
        validate(run)
        source(run/'state_signal.csv')
        raw = read_csv(run/'state_signal.csv')
        for r in calibration:
            if r['scheme'] != raw[0]['scheme']:
                continue
            sr = [s for s in raw if s['state_layer_name'] == r['state_layer']]
            assert len(sr) == 36
            assert math.isclose(pooled(sr, 'free_rms'), float(r['free_rms']), rel_tol=1e-12)
            assert math.isclose(pooled(sr, 'centered_half_difference_rms'), float(r['centered_half_difference_rms']), rel_tol=1e-12)
    references = {r['scheme']: r for r in calibration if r['layer'] == '4'}
    initial_estimates = []
    for parent in training:
        ref = references[parent['scheme']]
        ratio = float(ref['normalized_D_over_F'])*parent['beta']/float(ref['injected_beta'])
        initial_estimates.append(dict(parent, initial_output_D_over_F=ratio,
                                      method='linear projection from validated output-RMS-one calibration; not exact replay at this beta'))
    candidate_betas = []
    for scheme, ref in references.items():
        for target in [1., 10., 20.]:
            candidate_betas.append(dict(scheme=scheme, target_initial_output_D_over_F=target,
                                        estimated_beta=float(ref['injected_beta'])*target/ref['normalized_D_over_F'],
                                        corresponding_output_D=target*float(ref['free_rms']),
                                        method='linear projection; not a measured or training-qualified beta'))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/'training_outcomes.csv', training)
    write_csv(out/'final_phase_vs_performance.csv', phases)
    write_csv(out/'absolute_rms1_initialization.csv', calibration)
    write_csv(out/'initial_output_linear_estimates.csv', initial_estimates)
    write_csv(out/'candidate_output_betas.csv', candidate_betas)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), sharey=True, squeeze=False)
    for ri, sigma in enumerate([.0003, .0005]):
        for layer in ['Readout']:
            ax = axes[0, ri]
            for r in phases:
                if r['training_sigma'] != sigma or r['layer'] != layer:
                    continue
                color = COLORS[r['scheme']]
                marker = 'o' if r['family'] == 'p90' else 's'
                y = r['final_validation_pct']
                ax.plot([r['batch_ratio_p10'], r['batch_ratio_p90']], [y, y], color=color, alpha=.45, lw=2)
                ax.scatter(r['normalized_D_over_F'], y, color=color, marker=marker, s=50, zorder=3)
                if any(p['residual_failed_rows'] for p in phases if p['replay_run'] == r['replay_run']):
                    ax.scatter(r['normalized_D_over_F'], y, color='black', marker='x', s=22, zorder=4)
                ax.annotate(r['family']+' '+r['scheme'], (r['normalized_D_over_F'], y), xytext=(5, 5),
                            textcoords='offset points', fontsize=8, color=color)
            ax.set_xscale('log')
            ax.set_title(f'{layer} · training read noise σ={sigma:g}')
            ax.set_xlabel('Output centered displacement / free-state RMS (D/F)')
            ax.grid(alpha=.2)
            ax.margins(x=.3, y=.22)
            if ri == 0:
                ax.set_ylabel('Epoch-30 validation accuracy [%]')
    legend = [Line2D([0], [0], marker=m, linestyle='none', color='.3', label=l)
              for m, l in [('o', 'p90: A100 training'), ('s', 'p99: V100 training'), ('x', 'Any replay residual flag')]]
    fig.legend(handles=legend, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('Existing noisy Conv3 training: normalized OUTPUT displacement at its training beta\n'
                 'Points: pooled D/F; bars: batch 10–90% range · one seed · observational, not matched-displacement training', fontsize=11)
    fig.subplots_adjust(left=.065, right=.99, bottom=.19, top=.78, wspace=.13)
    for ext in ['png', 'pdf']:
        fig.savefig(out/f'normalized_displacement_vs_accuracy.{ext}', dpi=180)
    plt.close(fig)
    validation = dict(training_runs=11, replay_runs=8, phase_rows=32, validation_errors=[],
                      validated_bundles=validated, input_sha256=sources, cohort_sha256=sorted(cohort_ids),
                      analyzer_sha256=sha256_file(Path(__file__)), no_gpu_work=True,
                      official_test_read=False, optimizer_steps_applied=False,
                      historical_D_over_P_available=False,
                      calibration_table_evidence='three calibration bundles validate; summary matches pooled raw state RMS',
                      initial_output_projection='linear approximation; future beta choices are not measured or qualified',
                      clean_training_displacement='not supplied by these historical replay sources')
    atomic_write_json(out/'validation.json', validation)
    print(json.dumps(dict(output=str(out.relative_to(ROOT)), training_runs=11, replay_runs=8, phase_rows=32), indent=2))


if __name__ == '__main__':
    main()
