"""Validate and summarize the six refined-beta Conv3 training outcomes."""
import csv
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.analyze_layerwise_beta_training import epoch_rows, validate_contract
from experiments.reporting import validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'results/eqprop-conv3-refined-beta-training-20260919-v1'
REPORT = ROOT / 'paper_ready_results/conv3_refined_beta_training_20260919.md'


def without_dataset_root(config):
    value = copy.deepcopy(config)
    for dataset in value["datasets"].values():
        dataset["params"].pop("root", None)
    return value


def analyze():
    selections = list(csv.DictReader((REPORT.parent / 'beta_refinement_20260919_selections.csv').open()))
    rows = []
    for case in json.loads((STUDY / 'cases.json').read_text()):
        row = dict(case)
        config = ROOT / case['config']
        assert hashlib.sha256(config.read_bytes()).hexdigest() == case['config_sha256']
        cfg = json.loads(config.read_text())
        selected = next(s for s in selections if s['architecture'] == 'conv3' and s['scheme'] == case['scheme'] and float(s['threshold']) == case['threshold'])
        assert float(selected['beta']) == case['beta']
        row['minimum_cosine'] = float(selected['minimum_cosine'])
        row['maximum_norm_mismatch'] = float(selected['maximum_norm_mismatch'])
        control = ROOT / case['comparison_control']
        assert not validate_run(control)
        ccfg = json.loads((control / 'config.used.json').read_text())
        validate_contract(cfg, ccfg)
        assert without_dataset_root(cfg)["datasets"] == without_dataset_root(ccfg)["datasets"]
        assert cfg["energy_minimizer"] == ccfg["energy_minimizer"]
        cm = json.loads((control / 'metrics.json').read_text())
        ce = epoch_rows(control)
        assert len(ce) == 10
        row.update(control_beta=ccfg['eqprop']['injected_beta_B'], control_epochs=ce,
                   control_final10_pct=100*ce[-1]['metrics']['validation_accuracy'])
        parent = 'production' if case['array_index'] == 0 else 'retry-v2'
        task = STUDY / 'jz' / parent / f"task_{case['array_index']}"
        bundles = list((task / 'runs').glob(f"000_{case['case']}_*"))
        assert len(bundles) <= 1
        row.update(state='pending', epochs=[], epochs_completed=0)
        if bundles:
            run = bundles[0]
            assert not validate_run(run), run
            used = json.loads((run / 'config.used.json').read_text())
            validate_contract(used, cfg)
            assert without_dataset_root(used) == without_dataset_root(cfg)
            assert used['eqprop']['injected_beta_B'] == case['beta']
            assert used['lab']['epochs'] == 10
            status = json.loads((run / 'status.json').read_text())
            epochs = epoch_rows(run)
            assert [e['epoch'] for e in epochs] == list(range(1, len(epochs)+1))
            row.update(bundle=str(run.relative_to(ROOT)), state=status['state'], epochs=epochs, epochs_completed=len(epochs))
            if status['state'] == 'complete':
                assert len(epochs) == 10
                assert (task / 'worker_started/exit_code').read_text().strip() == '0'
                summary = json.loads((task / 'summary.json').read_text())
                assert len(summary) == 1 and summary[0]['returncode'] == 0 and not summary[0]['smoke']
                assert summary[0]['config_sha256'] == case['config_sha256']
                metrics = json.loads((run / 'metrics.json').read_text())
                assert metrics['official_test_evaluations'] == 0
                assert metrics['initial_parameter_state_sha256'] == cm['initial_parameter_state_sha256']
                for k in ('train_indices_sha256', 'validation_indices_sha256', 'first_epoch_batch_order_sha256'):
                    assert metrics['dataset_provenance'][k] == cm['dataset_provenance'][k]
                assert metrics['dataset_provenance']['train_batch_order_sha256'] == cm['dataset_provenance']['train_batch_order_sha256'][:10]
                histories = {k:np.load(run/f'{k}.npy', allow_pickle=False) for k in ('accuracy_train','accuracy_test','loss_train','loss_test')}
                assert all(len(v) == 10 and np.isfinite(v).all() for v in histories.values())
                values = histories['accuracy_test']
                assert np.allclose(values,[e['metrics']['validation_accuracy'] for e in epochs],rtol=0,atol=1e-12)
                import torch
                for f in ('best_model.pt','final_model.pt'):
                    ck = torch.load(run/f,map_location='cpu',weights_only=True)
                    for spec,t in zip(ck['schema'],ck['states'],strict=True):
                        assert t.dtype == torch.float64 and torch.isfinite(t).all()
                        if spec['name'].strip().startswith('Bias_'):
                            assert torch.count_nonzero(t) == 0
                row.update(final10_pct=100*float(values[-1]),best10_pct=100*float(values.max()),
                           early_stable=float(values.max()-values[-1]) < .05-1e-12,
                           delta_vs_control_pp=100*float(values[-1])-row['control_final10_pct'],
                           initializer_and_order_match=True)
            elif status['state'] == 'failed':
                row['failure'] = status
        rows.append(row)
    completed = sum(r['state'] == 'complete' for r in rows)
    terminal = sum(r['state'] in ('complete','failed') for r in rows)
    comparisons = []
    for scheme in ('baseline','ours','legacy'):
        low = next(r for r in rows if r['scheme'] == scheme and r['threshold'] == .95)
        high = next(r for r in rows if r['scheme'] == scheme and r['threshold'] == .90)
        pair = dict(scheme=scheme,beta95=low['beta'],beta90=high['beta'])
        if low['state'] == high['state'] == 'complete':
            pair['larger_minus_smaller_pp'] = high['final10_pct']-low['final10_pct']
        comparisons.append(pair)
    payload = dict(updated_at=datetime.now(timezone.utc).isoformat(),expected=6,completed=completed,terminal=terminal,rows=rows,comparisons=comparisons)
    (STUDY/'analysis.json').write_text(json.dumps(payload,indent=2)+'\n')
    fields=['scheme','threshold','beta','minimum_cosine','maximum_norm_mismatch','state','epochs_completed','final10_pct','best10_pct','early_stable','control_beta','control_final10_pct','delta_vs_control_pp','bundle']
    with REPORT.with_suffix('.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    with REPORT.with_name(REPORT.stem+'_epochs.csv').open('w') as f:
        fields=['scheme','role','beta','epoch','train_accuracy','validation_accuracy','train_loss','validation_loss'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        seen=set()
        for r in rows:
            series=[('candidate',r['beta'],r['epochs'])]
            if r['scheme'] not in seen:
                seen.add(r['scheme']);series.append(('historical_first10',r['control_beta'],r['control_epochs']))
            for role,beta,epochs in series:
                for e in epochs:w.writerow(dict(scheme=r['scheme'],role=role,beta=beta,epoch=e['epoch'],**{k:e['metrics'][k] for k in fields[-4:]}))
    text=['# Conv3 refined-beta training: ten-epoch comparison','',f"Updated {payload['updated_at']}. **{completed}/6 completed, {terminal}/6 terminal.**",'',
          'Exploratory, seed0, ordinary-MNIST fixed55k/5k validation, zero read noise, T=K=8, float64 centered EqProp, fixed scheme-specific Adam rates, matched initializer and minibatch order. Official-test evaluations remain zero.',
          'Select the largest measured beta whose cosine exceeds the listed threshold for every matrix on each of36 batches at both initialization and the saved BPTT checkpoint. The refined passing/failing beta brackets are within5%. No gradient-norm gate is imposed.', '',
          '| Scheme | Threshold | Injected beta | Worst cosine | State / epochs | Final / best validation | Delta vs old beta (pp) |',
          '|---|---:|---:|---:|---|---|---:|']
    for r in rows:
        acc=f"{r['final10_pct']:.2f}% / {r['best10_pct']:.2f}%" if r['state']=='complete' else '—'
        delta=f"{r['delta_vs_control_pp']:+.2f}" if r['state']=='complete' else '—'
        text.append(f"| {r['scheme']} | {r['threshold']:.2f} | {r['beta']:.12g} | {r['minimum_cosine']:.6f} | {r['state']} / {r['epochs_completed']} | {acc} | {delta} |")
    text+=['','| Scheme | Larger (.90) minus smaller (.95), final validation pp |','|---|---:|']
    for c in comparisons:
        v=f"{c['larger_minus_smaller_pp']:+.2f}" if 'larger_minus_smaller_pp' in c else 'Pending'
        text.append(f"| {c['scheme']} | {v} |")
    if completed == 6:
        text += ['', '**All six settings complete ten finite epochs and pass the final-drop screen.** '
                 'Larger-minus-smaller final validation is baseline +0.02pp, ours +0.04pp and legacy -0.04pp. '
                 'These one-seed differences show no consistent performance penalty from the larger p90 beta; '
                 'they do not establish statistical equivalence or a universal stability guarantee.',
                 'The p90 settings are admitted to the separately authorized thirty-epoch V100/A100 read-noise sweep. '
                 'That follow-up does not retroactively make these ten-epoch clean pilots full-horizon evidence.',
                 '[Closeout validation and accounting](../results/eqprop-conv3-refined-beta-training-20260919-v1/closeout-validation.json)']
    text+=['','Previous-beta controls use only the first ten epochs of the named historical30-epoch runs: '+ '; '.join(f"{r['scheme']} beta{r['control_beta']:g}: {r['control_final10_pct']:.2f}%" for r in rows if r['threshold']==.95)+'.',
           '', 'Stability requires ten finite epochs and final validation less than5pp below the run’s own best; this only screens gross collapse. Single-seed differences do not establish a reproducible performance advantage. No full30-epoch, noisy-training or multiseed qualification is implied.',
           '', 'Operational provenance: original2178212_0 is retained; original indices1–5 exited before training because the wrapper and runner both selected by Slurm array index. The wrapper fix explicitly selects exact-run index0 after choosing the config. Recovery canary2178310 passed, then array2178358 retried only those five cases. Failed startup attempts and smokes remain preserved and excluded. All accepted runs use Jean Zay H100 and unchanged scientific source/configs.',
           '', '[Validation curves](conv3_refined_beta_training_20260919.png) · [Summary CSV](conv3_refined_beta_training_20260919.csv) · [Epoch CSV](conv3_refined_beta_training_20260919_epochs.csv) · [Plan](../docs/eqprop_conv3_refined_beta_training_plan_20260919.md) · [Raw evidence](../results/eqprop-conv3-refined-beta-training-20260919-v1/)']
    REPORT.write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for ax,scheme in zip(axs,('baseline','ours','legacy')):
        subset=[r for r in rows if r['scheme']==scheme];r=subset[0]
        ax.plot(range(1,11),[100*e['metrics']['validation_accuracy'] for e in r['control_epochs']],'k--',label=f"Old beta {r['control_beta']:g}")
        for r in subset:
            ax.plot([e['epoch'] for e in r['epochs']],[100*e['metrics']['validation_accuracy'] for e in r['epochs']],'o-',ms=3,label=f"cos>{r['threshold']:.2f}, beta {r['beta']:.4g}")
        ax.set(title=scheme,xlabel='Epoch',ylabel='Validation accuracy (%)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Conv3 refined-beta training — seed 0, zero read noise')
    fig.savefig(REPORT.with_suffix('.png'),dpi=170);fig.savefig(REPORT.with_suffix('.pdf'));plt.close(fig)
    print(json.dumps(dict(completed=completed,terminal=terminal,comparisons=comparisons)))


if __name__ == '__main__':
    analyze()
