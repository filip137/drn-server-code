"""Validate collected CIFAR L8 batch sweep and summarize its frozen ranking."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments import reporting


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);args=parser.parse_args();root=args.root
    rows=[];validation={};curves={};initial_hashes=set();split_hashes=set()
    for batch in [16,32,64]:
        run=root/f'bs{batch}'
        errors=reporting.validate_run(run);validation[run.name]=errors
        if errors:raise ValueError(errors)
        manifest=json.loads((run/'manifest.json').read_text())
        result=json.loads((run/'result.json').read_text())
        metrics=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
        assert [m['epoch'] for m in metrics]==[1,2,3,4,5]
        assert manifest['dataset']['official_test_read'] is False
        assert result['completion']['official_test_read'] is False
        assert all(m['train_examples']==45000 and m['validation_examples']==5000 for m in metrics)
        assert metrics[-1]['steps']==5*math.ceil(45000/batch)
        assert all(len(m['first_batch_gradient_norms'])==19 and all(math.isfinite(v) and v>0 for v in m['first_batch_gradient_norms'].values()) for m in metrics)
        launcher='launcher-nom' if batch==32 and (root/'launcher-nom').exists() else 'launcher'
        assert (root/launcher/f'bs{batch}_exit_code').read_text().strip()=='0'
        assets=json.loads((run/'artifacts/assets.json').read_text())
        initial_hashes.add(assets['initial_sha256']);split_hashes.add(assets['split_sha256'])
        initial=torch.load(run/'checkpoints/initial_model.pt',map_location='cpu',weights_only=False)['model']
        checkpoint=torch.load(run/'checkpoints/final_model.pt',map_location='cpu',weights_only=False)
        assert checkpoint['epoch']==5
        final=checkpoint['model']
        bn_changes={}
        for section in final.values():
            assert all(torch.isfinite(v).all() for v in section.values())
        for name,value in final['module'].items():
            if name.endswith('num_batches_tracked'):assert value.item()==metrics[-1]['steps']
            if name.startswith('bridges.') and name.endswith(('.weight','.bias')):
                bn_changes[name]=float((value-initial['module'][name]).abs().mean())
                assert bn_changes[name]>0
        for name,value in final['conductances'].items():
            assert not torch.equal(value,initial['conductances'][name]),name
            assert value.min()>=1e-7 and value.max()<=10
        result=result['terminal_metrics'];last=metrics[-1]
        row={'batch_size':batch,'target':manifest['runtime']['target'],
             'epoch5_validation_accuracy':last['validation_accuracy'],
             'epoch5_validation_cross_entropy':last['validation_loss'],
             'best_validation_accuracy':result['best_validation_accuracy'],
             'elapsed_seconds':result['elapsed_seconds'],
             'mean_epoch_seconds':float(np.mean([m['epoch_seconds'] for m in metrics])),
             'peak_reserved_gib':max(m['peak_memory_reserved_bytes'] for m in metrics)/2**30,
             'final_solver_audit_passed':result['final_solver_audit_passed'],
             'bn_mean_absolute_changes':bn_changes}
        rows.append(row);curves[batch]=metrics
        del initial,checkpoint,final
    assert len(initial_hashes)==len(split_hashes)==1
    for run in [p.parent for p in root.rglob('manifest.json') if p.parent.name not in ['bs16','bs32','bs64']]:
        validation[run.name]=reporting.validate_run(run)
        assert not validation[run.name]
    ranking=sorted(rows,key=lambda r:(-r['epoch5_validation_accuracy'],r['epoch5_validation_cross_entropy'],r['elapsed_seconds']))
    summary={'rows':rows,'observed_accuracy_winner':ranking[0]['batch_size'],
             'ranking':[r['batch_size'] for r in ranking],
             'all_final_solver_audits_passed':all(r['final_solver_audit_passed'] for r in rows),
             'official_test_read':False,'validation':validation,
             'ranking_rule':'epoch5_validation_accuracy_then_lower_cross_entropy_then_shorter_runtime',
             'evidence_class':'exploratory_single_seed_validation',
             'shared_initialization_sha256':next(iter(initial_hashes)),
             'shared_split_sha256':next(iter(split_hashes))}
    analysis=root/'analysis';analysis.mkdir(exist_ok=True)
    reporting.atomic_write_json(analysis/'summary.json',summary)
    reporting.atomic_write_json(root/'closeout_validation.json',{'passed':True,'bundles':validation})
    columns=[k for k in rows[0] if k!='bn_mean_absolute_changes']
    with (analysis/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for batch,metrics in curves.items():
        epochs=[m['epoch'] for m in metrics]
        axs[0].plot(epochs,[100*m['validation_accuracy'] for m in metrics],marker='o',label=f'Batch {batch}')
        axs[1].plot(epochs,[m['validation_loss'] for m in metrics],marker='o',label=f'Batch {batch}')
    for ax in axs:ax.set_xlabel('Epoch');ax.grid(alpha=.25);ax.legend();ax.set_xticks([1,2,3,4,5])
    axs[0].set_ylabel('Validation accuracy (%)');axs[1].set_ylabel('Validation cross-entropy')
    fig.suptitle('L8 analog convolutions and classifier — CIFAR-10, seed 0')
    fig.tight_layout();fig.savefig(analysis/'learning_curves.png',dpi=160);plt.close(fig)
    table='\n'.join(f"| {r['batch_size']} | {r['target']} | {r['epoch5_validation_accuracy']*100:.2f}% | {r['epoch5_validation_cross_entropy']:.4f} | {r['mean_epoch_seconds']:.1f}s | {r['peak_reserved_gib']:.2f} | {r['final_solver_audit_passed']} |" for r in rows)
    caution=('All final solver audits passed.' if summary['all_final_solver_audits_passed'] else
             'Some final solver audits failed. Observed accuracy ranking is provisional; failing arms are not solver-qualified recommendations.')
    (analysis/'report.md').write_text(f'''# CIFAR-10 L8 analog batch-size comparison

All three seed-0 runs completed five epochs. The frozen epoch-5 validation
accuracy ranking is **{summary['ranking']}**, with batch **{ranking[0]['batch_size']}**
the observed winner. {caution}

| Batch | Host | Epoch-5 accuracy | Epoch-5 CE | Mean epoch | Peak GiB | Final solver audit |
|---|---|---:|---:|---:|---:|---|
{table}

All eight convolutions and the dense 8192-to-10 classifier are analog resistive
layers; three block boundaries retain digital max-pool and trainable affine BN.
All BN scale/shift pairs and all nine conductance tensors changed. Cross-entropy
without label smoothing, crop/flip augmentation, shared transferred L12 Adam
rates, voltage/current amplification 4/0.25 and input gains initialized to 100.

The deterministic 45k/5k split, saved initialization, per-epoch sample ordering
and per-example augmentation draws were shared. Only training batch files were
read; no official-test access. Batch size changes optimizer-step count and BN
statistics, so this compares five-epoch training setups, not equal compute.
At the user's request, batch 32 was moved to Nom-cool-1's RTX 3090; batches 16/64
use Fifi's RTX 5090. PyTorch/CUDA/cuDNN versions also differ. Report accuracy
as a comparison of these realized runs, and do not attribute throughput or
small accuracy differences solely to batch size. Nom's first smoke failed
before an optimizer step because installed cuDNN library bytes did not match
package metadata. An isolated cuDNN 9.1 runtime and repeated smoke fixed it;
the failed attempt is retained and excluded from the scientific comparison.
One seed and five epochs do not establish a statistically robust or long-run
optimum. Digital L8's 92.62% test result used a different split and 50 epochs.

All collected canonical bundles validate locally, including preparation,
smokes and the preserved failed Nom smoke. Complete epoch/example coverage,
finite checkpoints/gradients, bounds,
BN update counts and launcher exits pass. See summary.json for details.
''')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
