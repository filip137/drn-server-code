"""Independently replay directed MNIST checkpoints and summarize all declared cells."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from labs.tools.train_directed_eqprop_mnist import write_json


def numpy_predict(weights, x, tolerance=1e-10):
    """Independent NumPy implementation: no Torch model/settler/metric reuse."""
    h=weights["input"].shape[0]
    n=2*h+weights["forward2"].shape[0]
    drive=np.broadcast_to(weights["bias"],(len(x),n)).copy()
    drive[:,:h]+=x@weights["input"].T
    u=drive.copy()
    for iteration in range(1000):
        # Clip exponent only beyond float64 overflow; typical states are much smaller.
        r=1/(1+np.exp(np.clip(2-4*u,-700,700)))
        new=drive.copy()
        new[:,:h]+=r[:,h:2*h]@weights["backward1"].T
        new[:,h:2*h]+=r[:,:h]@weights["forward1"].T+r[:,2*h:]@weights["backward2"].T
        new[:,2*h:]+=r[:,h:2*h]@weights["forward2"].T
        if np.max(np.abs(new-u))<=tolerance:
            logits=r[:,2*h:]@weights["readout"].T+weights["readout_bias"]
            return logits
        u=new
    raise RuntimeError("Independent NumPy equilibrium did not converge")


def numpy_evaluate(weights,x,y,tolerance):
    loss,correct=0.,0
    for start in range(0,len(x),512):
        logits=numpy_predict(weights,x[start:start+512],tolerance)
        labels=y[start:start+512]
        shifted=logits-logits.max(-1,keepdims=True)
        loss+=np.sum(np.log(np.exp(shifted).sum(-1))-shifted[np.arange(len(labels)),labels])
        correct+=np.sum(logits.argmax(-1)==labels)
    return dict(accuracy=float(correct/len(x)),loss=float(loss/len(x)),examples=len(x))


def verify(root):
    root=Path(root)
    run=json.loads((root/"run.json").read_text())
    status=json.loads((root/"status.json").read_text())
    if status["status"]!="complete":
        raise AssertionError(f"Expected completed experiment, got {status}")
    expected=run["cases"]
    actual={p.parent.name for p in root.glob("*/result.json")}
    assert actual==set(expected),(actual,set(expected))
    data={k:np.load(root/"data"/(k+".npy"),mmap_mode="r")
          for k in ("train_y","val_x","val_y","test_x","test_y")}
    rows=[]
    for name,cfg in expected.items():
        path=root/name
        result=json.loads((path/"result.json").read_text())
        assert result["status"]=="complete" and result["config"]==cfg
        history=json.loads((path/"metrics.json").read_text())
        assert [r["epoch"] for r in history]==list(range(1,cfg["epochs"]+1))
        weights=np.load(path/"final.npz")
        assert int(weights["epoch"])==cfg["epochs"]
        assert json.loads(str(weights["config_json"]))==cfg
        for p in ("input","forward1","forward2","backward1","backward2","bias","readout","readout_bias"):
            assert np.isfinite(weights[p]).all(),(name,p)
        norm=max(np.linalg.norm(np.concatenate((weights["backward1"],weights["forward2"])),ord=2),
                 np.linalg.norm(np.concatenate((weights["forward1"],weights["backward2"]),axis=1),ord=2))
        assert norm<=cfg["cap"]+1e-10
        # Stored datasets and all final examples are independently replayed.
        replay={}
        for split in ("val","test"):
            if split=="test" and not cfg["evaluate_test"]:
                assert result["final_test"] is None
                continue
            independent=numpy_evaluate(weights,data[split+"_x"],data[split+"_y"],cfg["tolerance"])
            recorded=result["final_validation" if split=="val" else "final_test"]
            assert independent["accuracy"]==recorded["accuracy"],(name,split)
            assert abs(independent["loss"]-recorded["loss"])<1e-9,(name,split)
            replay[split]=independent
        count=result["training_counts"]
        examples=cfg["epochs"]*len(data["train_y"])
        phases=(1 if cfg["method"]=="adjoint" else 9 if cfg["method"].startswith("learned")
                else 11 if "probe" in cfg["method"] else 3)
        assert count["equilibrations"]==examples*phases
        assert count["state_reads"]==examples*phases
        assert count["scalar_reads"]==examples*phases*(2*cfg["hidden"]+10)
        assert count["homeostasis_gaussian_vectors"]==(5*examples if "homeo" in cfg["method"] else 0)
        assert count["homeostasis_jvps"]==(10*examples if "homeo" in cfg["method"] else 0)
        assert count["max_residual"]<=cfg["tolerance"]
        if cfg["method"].startswith("learned"):
            assert int(weights["feedback_observations"])==4*examples
            assert weights["feedback_matrix"].shape==(2*cfg["hidden"]+10,10)
            assert np.isfinite(weights["feedback_matrix"]).all()
        if cfg["method"]!="adjoint":
            assert count["oracle_adjoint_iterations"]==0
        audits=json.loads((path/"audits.json").read_text())
        assert [a["epoch"] for a in audits]==([0,1] if cfg["epochs"]==1 else [0,1,cfg["epochs"]])
        # Verify initial/final symmetry scores independently from the saved matrices.
        for file_name,record in (("initial.npz",audits[0]),("final.npz",audits[-1])):
            state=np.load(path/file_name)
            h=cfg["hidden"]; n=2*h+10
            w=np.zeros((n,n))
            w[:h,h:2*h]=state["backward1"]
            w[h:2*h,:h]=state["forward1"]
            w[h:2*h,2*h:]=state["backward2"]
            w[2*h:,h:2*h]=state["forward2"]
            j=w-np.eye(n)
            sn,an=np.linalg.norm((j+j.T)/2),np.linalg.norm((j-j.T)/2)
            assert math.isclose(record["code_antisymmetry_norm"],an,abs_tol=1e-10)
            assert math.isclose(record["code_symmetry_score"],sn/(sn+an),abs_tol=1e-10)
        row=dict(case=name,method=cfg["method"],alpha=cfg["alpha"],seed=cfg["seed"],
                 validation_accuracy=result["final_validation"]["accuracy"],
                 test_accuracy=None if result["final_test"] is None else result["final_test"]["accuracy"],
                 elapsed_seconds=result["elapsed_seconds"],phases_per_example=phases,
                 training_equilibrations=count["equilibrations"],
                 projection_batches=sum(h["projection_batches"] for h in history),
                 initial_code_antisymmetry_norm=audits[0]["code_antisymmetry_norm"],
                 **{k:v for k,v in audits[-1].items() if k not in ("epoch","audit_counts")})
        rows.append(row)
        write_json(path/"independent_verification.json",dict(status="passed",replay=replay,
            count_checks=True,checkpoint_checks=True,symmetry_checks=True))
        print(f"verified {name}",flush=True)
    write_json(root/"verification.json",dict(status="passed",expected=len(expected),verified=len(rows),
        dataset="Full validation and final test cohorts when test evaluation is enabled",
        checks="Independent NumPy equilibrium/cost/predictions, parameter constraints, symmetry, phase counts, declared coverage"))
    return rows


def summarize(root,rows):
    root=Path(root)
    with (root/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups=[]
    for angle in sorted({r["alpha"] for r in rows}):
        for method in dict.fromkeys(r["method"] for r in rows):
            group=[r for r in rows if r["method"]==method and r["alpha"]==angle]
            entry=dict(alpha=angle,method=method,seeds=len(group))
            for key in ("validation_accuracy","test_accuracy","hidden1_feedback_cosine",
                        "input_gradient_cosine","code_antisymmetry_norm","code_symmetry_score",
                        "membrane_symmetry_score","elapsed_seconds"):
                values=[r[key] for r in group if r[key] is not None]
                if values:
                    entry[key+"_mean"]=float(np.mean(values))
                    entry[key+"_std"]=float(np.std(values))
            groups.append(entry)
    write_json(root/"aggregate.json",groups)
    methods=list(dict.fromkeys(r["method"] for r in rows))
    angles=sorted({r["alpha"] for r in rows})
    accuracy_key="test_accuracy" if rows[0]["test_accuracy"] is not None else "validation_accuracy"
    lines=["# Directed EqProp on MNIST", "",
        "Exploratory comparison, fixed final epoch. Values are mean ± population SD across seeds.", "",
        "| Method | "+" | ".join(f"Initial mixing {a:g}° (%)" for a in angles)+" | Training equilibrations / example |",
        "|---|"+"---:|"*(len(angles)+1)]
    for method in methods:
        cells=[]
        for angle in angles:
            g=next(g for g in groups if g["method"]==method and g["alpha"]==angle)
            cells.append(f"{100*g[accuracy_key+'_mean']:.2f} ± {100*g[accuracy_key+'_std']:.2f}")
        phases=next(r["phases_per_example"] for r in rows if r["method"]==method)
        lines.append("| "+method+" | "+" | ".join(cells)+f" | {phases} |")
    lines += ["", f"Metric: {accuracy_key.replace('_',' ')}. Full coverage: {len(rows)} trajectories.", "",
        "Homeostasis adds five Gaussian vectors, ten forward JVPs and parameter AD per example. "
        "The adjoint reference additionally uses digital transposed-weight iterations. Phase counts "
        "do not equate these digital operations with physical equilibrations.", "",
        "The membrane VF control measures membrane changes under the membrane cost gradient. "
        "Activity EP measures activated-state changes under the activity cost gradient, matching "
        "the homeostasis implementation's convention. The main code symmetry score refers to W-I; "
        "the actual membrane Jacobian W D-I is recorded separately.", "",
        "Four-probe residual estimates use either measured activity EP or the learned feedback "
        "predictor as baseline. Learned predictors are updated only after forming the current "
        "gradient; audits do not train them. All network parameters use the same Adam settings.", "",
        "![Final comparisons](comparison.png)", "", "![Validation learning curves](learning_curves.png)", "",
        "All final validation and enabled test predictions/losses were independently replayed "
        "with NumPy. Coverage, phase counts, finite weights, recurrent norm bounds and symmetry "
        "metrics were checked. See verification.json, summary.csv and per-case artifacts.", "",
        "Limits: two small hidden layers, a recurrent contraction constraint, five training epochs "
        "in the main run, two seeds, no read noise, and a common learning rate without tuning. "
        "These are mechanism comparisons, not a reproduction of published accuracy or a hardware speed claim.", ""]
    (root/"report.md").write_text("\n".join(lines))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    metrics=(("test_accuracy" if rows[0]["test_accuracy"] is not None else "validation_accuracy","Accuracy"),
             ("hidden1_feedback_cosine","First hidden-layer feedback cosine"),
             ("input_gradient_cosine","Input-weight task-gradient cosine"),
             ("code_antisymmetry_norm","Absolute antisymmetry of W-I"),
             ("code_symmetry_score","Symmetry score of W-I"),
             ("membrane_symmetry_score","Symmetry score of W D-I"))
    for ax,(key,title) in zip(axes.flat,metrics):
        for method in methods:
            group=[next(g for g in groups if g["method"]==method and g["alpha"]==a) for a in angles]
            ax.errorbar(angles,[g[key+"_mean"] for g in group],
                yerr=[g[key+"_std"] for g in group],marker="o",capsize=2,label=method)
        ax.set_title(title); ax.set_xlabel("Initial mixing angle (degrees)"); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle("Directed EqProp on MNIST — final epoch, mean ± population SD across seeds")
    fig.savefig(root/"comparison.png",dpi=180)
    fig.savefig(root/"comparison.pdf")
    plt.close(fig)
    fig,axes=plt.subplots(1,len(angles),figsize=(5*len(angles),4),squeeze=False,constrained_layout=True)
    for ax,angle in zip(axes.flat,angles):
        for method in methods:
            cohort=[r for r in rows if r["alpha"]==angle and r["method"]==method]
            curves=[json.loads((root/r["case"]/"metrics.json").read_text()) for r in cohort]
            values=np.array([[e["validation"]["accuracy"] for e in c] for c in curves])
            epochs=np.arange(1,values.shape[1]+1)
            avg,sd=values.mean(0),values.std(0)
            ax.plot(epochs,avg,label=method)
            ax.fill_between(epochs,avg-sd,avg+sd,alpha=.12)
        ax.set_title(f"Initial mixing angle {angle:g}°"); ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation accuracy"); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(root/"learning_curves.png",dpi=180)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    rows=verify(args.root)
    summarize(args.root,rows)


if __name__=="__main__":
    main()
