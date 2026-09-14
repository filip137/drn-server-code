"""Verify saved Fashion-MNIST runs by independent NumPy replay and plot results."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, logsumexp

from labs.tools.train_directed_eqprop_mnist import write_json, utc


LABELS={"vf_ad":"VF (exact response)","vf_ad_homeo":"VF + homeostasis",
        "vf_central":"VF (paired nudges)","zo_learned4":"Learned baseline + 4 probes"}
COLORS={"vf_ad":"#5b6470","vf_ad_homeo":"#19794f","vf_central":"#315ca8","zo_learned4":"#bd5126"}


def replay(weights, x, y, steps=150, batch_size=500):
    """Separate implementation: NumPy/SciPy only, no model/solver imports."""
    h=weights["forward1"].shape[0]
    n=2*h+weights["forward2"].shape[0]
    total,correct,residual=0.,0,0.
    residuals,period_two,odd_predictions=[],0,0
    for start in range(0,len(x),batch_size):
        xb,yb=x[start:start+batch_size],y[start:start+batch_size]
        drive=np.broadcast_to(weights["bias"],(len(xb),n)).copy()
        drive[:,:h]+=xb@weights["input"].T
        u=np.zeros_like(drive)
        def step(u):
            r=expit(4*u-2)
            return np.concatenate((r[:,h:2*h]@weights["backward1"].T,
                r[:,:h]@weights["forward1"].T+r[:,2*h:]@weights["backward2"].T,
                r[:,h:2*h]@weights["forward2"].T),axis=1)+drive
        for _ in range(steps):
            u=step(u)
        u_next=step(u)
        norms=np.linalg.norm(u_next-u,axis=1)
        residuals.append(norms)
        residual=max(residual,float(norms.max()))
        period_two+=int(((norms>1e-3)&(np.linalg.norm(step(u_next)-u,axis=1)<1e-3)).sum())
        r=expit(4*u-2)
        logits=r[:,2*h:]@weights["readout"].T+weights["readout_bias"]
        total+=float(np.sum(logsumexp(logits,axis=1)-logits[np.arange(len(yb)),yb],dtype=np.float64))
        correct+=int((logits.argmax(-1)==yb).sum())
        odd_logits=expit(4*u_next[:,2*h:]-2)@weights["readout"].T+weights["readout_bias"]
        odd_predictions+=int((odd_logits.argmax(-1)!=logits.argmax(-1)).sum())
    norms=np.concatenate(residuals)
    return dict(accuracy=correct/len(x),loss=total/len(x),examples=len(x),free_residual_max=residual,
        convergence_threshold_l2=1e-3,nonconverged_examples=int((norms>1e-3).sum()),
        nonconverged_fraction=float((norms>1e-3).mean()),period_two_examples=period_two,
        prediction_changes_after_one_step=odd_predictions,
        residual_l2_quantiles={str(q):float(np.quantile(norms,q)) for q in (.5,.9,.95,.99,1.)})


def stable_cohort_audit(root,name,cfg):
    """Post-hoc validity audit: restrict force-adjoint metrics to free equilibria.

    Checkpoint and classifier accuracy stay fixed. An inverse at an oscillating
    state must not be labeled the equilibrium task gradient.
    """
    import torch
    from labs.fmnist_homeostasis import PaperNet,MeasuredFeedback,free_state,diagnostics
    torch.set_num_threads(1)
    model=PaperNet(hidden=cfg['hidden']).double()
    with np.load(root/name/'final.npz') as f:
        model.load_state_dict({k:torch.tensor(f[k]) for k in model.state_dict()})
        learner=None
        if cfg['method']=='zo_learned4':
            learner=MeasuredFeedback(model.n,model.outputs)
            learner.matrix=f['feedback_matrix'].copy()
    x=torch.tensor(np.load(root/'data/train_x.npy',mmap_mode='r')[:8]).double()
    y=torch.tensor(np.load(root/'data/train_y.npy',mmap_mode='r')[:8])
    with torch.no_grad():
        u,drive=free_state(model,x,cfg['free_steps'])
        norms=model.force(u,drive).norm(dim=-1)
        valid=norms<1e-7
    audit=None if not bool(valid.any()) else diagnostics(model,x[valid],y[valid],cfg['method'],learner,
        beta=cfg['beta'],free_steps=cfg['free_steps'],response_steps=cfg['response_steps'])
    return dict(status='posthoc validity audit',cohort='first eight training examples, as in training audits',
        valid_examples=int(valid.sum()),excluded_examples=int((~valid).sum()),
        valid_cohort_indices=valid.nonzero().flatten().tolist(),free_residuals=norms.tolist(),
        threshold_l2=1e-7,stable_only=audit)


def verify_case(root, name, cfg, train_examples):
    path=root/name
    result=json.loads((path/"result.json").read_text())
    status=json.loads((path/"status.json").read_text())
    metrics=json.loads((path/"metrics.json").read_text())
    audits=json.loads((path/"audits.json").read_text())
    assert result["status"]==status["status"]=="complete",name
    assert result["config"]==cfg,name
    assert [r["epoch"] for r in metrics]==list(range(1,cfg["epochs"]+1)),name
    assert audits[-1]["epoch"]==cfg["epochs"],name
    work=result["training_work"]
    examples=cfg["epochs"]*train_examples
    phase_multiplier={"vf_ad":0,"vf_ad_homeo":0,"vf_central":2,"zo_learned4":8}[cfg["method"]]
    response_multiplier=1 if cfg["method"].startswith("vf_ad") else phase_multiplier
    assert work["examples"]==work["free_phases"]==examples,name
    assert work["measured_nudge_phases"]==phase_multiplier*examples,name
    assert work["state_iterations"]==(cfg["free_steps"]+response_multiplier*cfg["response_steps"])*examples,name
    assert work["tangent_jvps"]==(cfg["response_steps"]*examples if cfg["method"].startswith("vf_ad") else 0),name
    assert work["homeostasis_jvps"]==(10*examples if cfg["method"]=="vf_ad_homeo" else 0),name
    assert work["homeostasis_gaussian_vectors"]==(5*examples if cfg["method"]=="vf_ad_homeo" else 0),name
    with np.load(path/"final.npz") as f:
        w={k:f[k] for k in f.files}
    assert int(w["epoch"])==cfg["epochs"],name
    assert json.loads(str(w["config_json"]))==cfg,name
    params=("input","forward1","forward2","backward1","backward2","bias","readout","readout_bias")
    assert all(np.isfinite(w[k]).all() for k in params),name
    with np.load(path/"initial.npz") as initial:
        assert all(not np.array_equal(w[k],initial[k]) for k in params),name
    if cfg["method"]=="zo_learned4":
        assert int(w["feedback_observations"])==4*examples,name
        assert np.isfinite(w["feedback_matrix"]).all(),name
        assert work["probe_scalar_reads"]==8*examples,name
        assert np.isclose(work["probe_excitation_sq"],8*examples*cfg["beta"]**2),name
    cache=root/"data"
    if cfg["no_test"]:
        x,y=np.load(cache/"train_x.npy",mmap_mode="r")[:200],np.load(cache/"train_y.npy",mmap_mode="r")[:200]
    else:
        x,y=np.load(cache/"test_x.npy",mmap_mode="r"),np.load(cache/"test_y.npy",mmap_mode="r")
    independent=replay(w,x,y,cfg["free_steps"])
    reported=result["final_evaluation"]
    # Different float32 BLAS/sigmoid implementations may change tied predictions.
    # Record the exact discrepancy and never silently substitute one accuracy.
    accuracy_delta=independent["accuracy"]-reported["accuracy"]
    loss_delta=independent["loss"]-reported["loss"]
    assert abs(accuracy_delta)<=1/len(y)+1e-12,(name,independent,reported)
    assert abs(loss_delta)<2e-5,(name,independent,reported)
    stable_audit=stable_cohort_audit(root,name,cfg)
    write_json(path/'stable_cohort_audit.json',stable_audit)
    return dict(case=name,status="verified",independent=independent,stable_cohort_audit=stable_audit,
        accuracy_difference=accuracy_delta,loss_difference=loss_delta,
        checkpoint_sha256=hashlib.sha256((path/"final.npz").read_bytes()).hexdigest()),result,metrics,audits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    args=p.parse_args()
    root=args.root
    run=json.loads((root/"run.json").read_text())
    dataset=json.loads((root/"dataset.json").read_text())
    assert dataset["dataset"]=="Fashion-MNIST"
    status=json.loads((root/"status.json").read_text())
    assert status["status"]=="complete" and status["completed"]==len(run["cases"])
    for name,digest in run["source_sha256"].items():
        assert hashlib.sha256((root/"source"/name).read_bytes()).hexdigest()==digest,name
    # Methods must have exactly the same initial model for each seed.
    initial_by_seed={}
    for name,cfg in run["cases"].items():
        with np.load(root/name/"initial.npz") as f:
            params={k:f[k] for k in ("input","forward1","forward2","backward1","backward2","bias","readout","readout_bias")}
        seed=cfg["seed"]
        if seed in initial_by_seed:
            assert all(np.array_equal(v,initial_by_seed[seed][k]) for k,v in params.items()),name
        initial_by_seed[seed]=params
    receipts,by_method=[],defaultdict(list)
    for name,cfg in run["cases"].items():
        receipt,result,metrics,audits=verify_case(root,name,cfg,dataset["train_examples"])
        receipts.append(receipt)
        by_method[cfg["method"]].append(dict(result=result,metrics=metrics,audits=audits,receipt=receipt))
        write_json(root/"verification_progress.json",dict(verified=len(receipts),expected=len(run["cases"]),last=receipt,utc=utc()))
        print(f"Verified {name}: {result['final_evaluation']['accuracy']:.4f}, replay delta={receipt['accuracy_difference']:g}",flush=True)
    summary={}
    for method,runs in by_method.items():
        accuracies=np.array([r["result"]["final_evaluation"]["accuracy"]*100 for r in runs])
        stable=[r['receipt']['stable_cohort_audit']['stable_only'] for r in runs
                if r['receipt']['stable_cohort_audit']['stable_only'] is not None]
        summary[method]=dict(seeds=len(runs),test_accuracy_mean=float(accuracies.mean()),
            test_accuracy_sample_sd=float(accuracies.std(ddof=1)) if len(runs)>1 else 0.,
            accuracy_by_seed={str(r["result"]["config"]["seed"]):r["result"]["final_evaluation"]["accuracy"]*100 for r in runs},
            mean_elapsed_seconds=float(np.mean([r["result"]["elapsed_seconds"] for r in runs])),
            mean_final_jacobian_symmetry=float(np.mean([r["audits"][-1]["jacobian_symmetry"] for r in runs])),
            mean_hidden1_feedback_cosine=float(np.mean([a["layer_feedback_cosine"]["hidden1"] for a in stable])) if stable else None,
            stable_feedback_seed_count=len(stable),
            excluded_feedback_examples=sum(r['receipt']['stable_cohort_audit']['excluded_examples'] for r in runs),
            mean_nonconverged_test_fraction=float(np.mean([r['receipt']['independent']['nonconverged_fraction'] for r in runs])),
            nonconverged_test_fraction_by_seed={str(r['result']['config']['seed']):r['receipt']['independent']['nonconverged_fraction'] for r in runs},
            max_free_horizon_difference=max(r["audits"][-1]["free_150_vs_300_max"] for r in runs),
            max_response_horizon_relative_error=max(r["audits"][-1]["response_20_vs_40_relative_error"] for r in runs),
            max_measured_projection_relative_error=max(r["audits"][-1].get("measured_probe_projection_relative_error",0) for r in runs))
    write_json(root/"summary.json",summary)
    epochs=next(iter(run["cases"].values()))["epochs"]
    fig,axes=plt.subplots(1,3,figsize=(15,4.2))
    for method,runs in by_method.items():
        series=np.array([[100*(1-m["evaluation"]["accuracy"]) for m in r["metrics"]] for r in runs])
        xs=np.arange(1,epochs+1)
        mean=series.mean(0); sd=series.std(0,ddof=1) if len(runs)>1 else np.zeros(epochs)
        axes[0].plot(xs,mean,label=LABELS[method],color=COLORS[method])
        axes[0].fill_between(xs,mean-sd,mean+sd,color=COLORS[method],alpha=.13)
        work_x=np.array([m["training_work"]["state_iterations"] for m in runs[0]["metrics"]])/1e6
        axes[1].plot(work_x,mean,label=LABELS[method],color=COLORS[method])
        audits=runs[0]["audits"]
        axs=[a["epoch"] for a in audits]
        sym=np.array([[a["jacobian_symmetry"] for a in r["audits"]] for r in runs])
        axes[2].plot(axs,sym.mean(0),label=LABELS[method],color=COLORS[method],marker='o')
    axes[0].set(xlabel="Epoch",ylabel="Official test error (%)",title=f"{epochs}-epoch Fashion-MNIST comparison")
    axes[1].set(xlabel="Training state iterations (millions)",ylabel="Official test error (%)",title="State-update budget (JVP cost separate)")
    axes[2].set(xlabel="Epoch",ylabel="||S|| / (||S|| + ||A||)",title="Paper-code Jacobian symmetry")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root/"comparison.png",dpi=180)
    fig.savefig(root/"comparison.pdf")
    plt.close(fig)
    lines=["# Fashion-MNIST: shortened Figure 4a-d comparison", "",
        f"Verified {len(receipts)} trajectories, {epochs} epochs; 256+256 hidden units, untied weights, Adam 1e-4, batch 50.",
        "Paper-style 60,000/10,000 split and unit-scaled pixels. Fixed final epoch, no test-based selection.","",
        "| Method | Test accuracy, mean ± sample SD (%) | Jacobian symmetry | Hidden-1 feedback cosine |",
        "|---|---:|---:|---:|"]
    for method,s in summary.items():
        cosine='unavailable' if s['mean_hidden1_feedback_cosine'] is None else f"{s['mean_hidden1_feedback_cosine']:.3f}"
        lines.append(f"| {LABELS[method]} | {s['test_accuracy_mean']:.2f} ± {s['test_accuracy_sample_sd']:.2f} | {s['mean_final_jacobian_symmetry']:.3f} | {cosine} |")
    lines += ["", "Feedback cosines above exclude non-equilibrated examples from the fixed eight-example cohort. The probe cosine uses a single noisy four-probe realization; it is not the cosine of an averaged training gradient.",
        "", "**Convergence limitation:** some test states have force residual norm above 1e-3 after the published 150 steps. The reported accuracy is for that fixed-step classifier. An inverse force Jacobian at a non-equilibrium state is not an equilibrium task gradient; those examples are excluded from the feedback-angle table. The full-run accuracy comparison alone cannot establish the equilibrium correction's benefit.",
        "", "## Limits and convergence", "",
        "This is a PyTorch equation-level replication shortened from 50 to 10 epochs by user request. RNG and data ordering differ from the authors' JAX/TensorFlow implementation. The measured predictor/correction is our addition. These endpoints do not reproduce the paper's final 50-epoch numbers.",
        "", "Every main training phase uses its fixed paper step budget. Counts describe phases and iterations, not certified equilibrations. Float64 read-only audits check the exact force adjoint and doubled horizons; float32 test replay checks saved predictions independently.", ""]
    for method,s in summary.items():
        lines.append(f"- {LABELS[method]}: {100*s['mean_nonconverged_test_fraction']:.2f}% of test examples fail the residual threshold on average; {s['excluded_feedback_examples']} of {8*s['seeds']} feedback-audit examples excluded. Original unfiltered audit: maximum free-horizon change {s['max_free_horizon_difference']:.3g}; VF response 20-vs-40 relative discrepancy {s['max_response_horizon_relative_error']:.3g}. See stable_cohort_audit.json for equilibrium-only results.")
    lines += ["", "## Cost accounting", "",
        "Per example: VF requires 170 state iterations plus 20 tangent JVPs; homeostasis additionally uses 10 JVPs and parameter AD. Paired VF requires 190 state iterations. Learned four-probe feedback requires 310 state iterations and eight measured nudge phases, with no differentiation through the recurrent dynamics. Its predictor uses known local activation slopes, and the local synaptic parameter-force derivatives remain known. Each has one free phase. These are different operations; a state-iteration plot is not an energy or total-compute comparison.",
        "", "All source snapshots, initial/final weights, per-epoch metrics, read-only audits and independent replay receipts are retained in this directory.", ""]
    (root/"report.md").write_text("\n".join(lines))
    write_json(root/"completion.json",dict(status="verified",expected=len(run["cases"]),verified=len(receipts),
        finished_utc=utc(),receipts=receipts,source_snapshot_hashes_verified=True,
        all_initial_models_matched_by_seed=True,all_work_counts_verified=True,
        replay_accuracy_tolerance="one image to accommodate float32 implementation ties; exact differences retained",
        replay_loss_tolerance=2e-5))


if __name__=="__main__":
    main()
