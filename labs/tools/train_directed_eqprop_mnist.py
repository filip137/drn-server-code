"""Exploratory full-MNIST comparison of VF EqProp, homeostasis and probe correction."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
import traceback

import numpy as np
import torch

from labs.directed_eqprop import (METHODS, Counts, DirectedNet, LearnedFeedback, audit, settle,
                                  training_gradient)
from labs.mnist_eqprop_data import load_mnist


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


@torch.no_grad()
def evaluate(model, x, y, tolerance, batch_size=512):
    correct, loss, counts = 0, 0., Counts()
    for start in range(0,len(x),batch_size):
        xb=torch.tensor(x[start:start+batch_size])
        yb=torch.tensor(y[start:start+batch_size])
        u=settle(model,model.drive(xb),tolerance=tolerance,counts=counts)
        losses,logits,_,_=model.cost(u,yb)
        loss+=float(losses.sum())
        correct+=int((logits.argmax(-1)==yb).sum())
    return dict(accuracy=correct/len(x),loss=loss/len(x),examples=len(x),counts=counts.__dict__)


def save_weights(path, model, cfg, epoch, learner=None):
    extra={} if learner is None else dict(feedback_matrix=learner.matrix,
                                          feedback_observations=learner.observations)
    np.savez(path, **{k:v.detach().numpy() for k,v in model.state_dict().items()},
             config_json=json.dumps(cfg),epoch=epoch,**extra)


def run_case(cfg, output, cache):
    torch.set_num_threads(1)
    path=Path(output)
    path.mkdir(parents=True)
    started=time.perf_counter()
    counts=Counts()
    metadata=dict(config=cfg,pid=os.getpid(),host=platform.node(),started_utc=utc())
    write_json(path/"config.json",metadata)
    metrics,audits=[],[]
    last_heartbeat=0.

    def heartbeat(state,epoch,batch=0):
        nonlocal last_heartbeat
        last_heartbeat=time.perf_counter()
        write_json(path/"status.json",dict(status=state,epoch=epoch,batch=batch,
            heartbeat_utc=utc(),elapsed_seconds=last_heartbeat-started,pid=os.getpid(),
            cpu_seconds=time.process_time(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            completed_epochs=len(metrics),training_counts=counts.__dict__))

    try:
        heartbeat("loading",0)
        data={name:np.load(Path(cache)/(name+".npy"),mmap_mode="r")
              for name in ("train_x","train_y","val_x","val_y","test_x","test_y")}
        model=DirectedNet(hidden=cfg["hidden"],seed=cfg["seed"],alpha=cfg["alpha"],cap=cfg["cap"])
        learner=LearnedFeedback(model.n,model.outputs) if cfg["method"].startswith("learned") else None
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"])
        order_rng=np.random.default_rng(41000+cfg["seed"])
        probe_rng=torch.Generator().manual_seed(51000+cfg["seed"])
        homeo_rng=torch.Generator().manual_seed(61000+cfg["seed"])
        audit_x=torch.tensor(data["train_x"][:16])
        audit_y=torch.tensor(data["train_y"][:16])
        feedback_cfg=dict(beta=cfg["beta"],tolerance=cfg["tolerance"],noise=cfg["noise"],learner=learner)
        save_weights(path/"initial.npz",model,cfg,0,learner)
        audits.append(dict(epoch=0,**audit(model,audit_x,audit_y,cfg["method"],**feedback_cfg)))
        write_json(path/"audits.json",audits)
        for epoch in range(1,cfg["epochs"]+1):
            epoch_start=time.perf_counter()
            loss,accuracy,examples=0.,0.,0
            projections,minimum_projection=0,1.
            homeo_sum,homeo_examples=0.,0
            order=order_rng.permutation(len(data["train_y"]))
            for batch,start in enumerate(range(0,len(order),cfg["batch_size"]),1):
                indices=order[start:start+cfg["batch_size"]]
                x=torch.from_numpy(data["train_x"][indices])
                y=torch.from_numpy(data["train_y"][indices])
                gradient,new_counts,stats=training_gradient(model,x,y,cfg["method"],probe_rng,
                    homeo_generator=homeo_rng,homeo_coefficient=cfg["homeo_coefficient"],**feedback_cfg)
                if not all(bool(torch.isfinite(g).all()) for g in gradient.values()):
                    raise RuntimeError("Non-finite training gradient")
                optimizer.zero_grad(set_to_none=True)
                for name,p in model.named_parameters():
                    p.grad=gradient[name]
                optimizer.step()
                factor=model.project()
                projections+=int(factor<1.-1e-12)
                minimum_projection=min(minimum_projection,factor)
                counts.add(new_counts)
                loss+=stats["loss"]*len(x)
                accuracy+=stats["accuracy"]*len(x)
                examples+=len(x)
                if "homeostasis_sample_loss" in stats:
                    homeo_sum+=stats["homeostasis_sample_loss"]*len(x)
                    homeo_examples+=len(x)
                if time.perf_counter()-last_heartbeat>20:
                    heartbeat("training",epoch,batch)
            validation=evaluate(model,data["val_x"],data["val_y"],cfg["tolerance"])
            record=dict(epoch=epoch,train_loss=loss/examples,train_accuracy=accuracy/examples,
                validation=validation,epoch_seconds=time.perf_counter()-epoch_start,
                elapsed_seconds=time.perf_counter()-started,training_counts=counts.__dict__.copy(),
                projection_batches=projections,minimum_projection_factor=minimum_projection)
            if homeo_examples:
                record["homeostasis_sample_loss"]=homeo_sum/homeo_examples
            metrics.append(record)
            write_json(path/"metrics.json",metrics)
            if epoch in (1,cfg["epochs"]):
                audits.append(dict(epoch=epoch,**audit(model,audit_x,audit_y,cfg["method"],**feedback_cfg)))
                write_json(path/"audits.json",audits)
            save_weights(path/"latest.npz",model,cfg,epoch,learner)
            # Complete epoch-boundary state retained for diagnosis/recovery.
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,
                config=cfg,probe_rng=probe_rng.get_state(),homeo_rng=homeo_rng.get_state(),
                order_rng=order_rng.bit_generator.state,counts=counts.__dict__,metrics=metrics,
                audits=audits,learner=None if learner is None else learner.__dict__),path/"latest.pt")
            heartbeat("training",epoch)
            print(f"{path.name} epoch={epoch} val={validation['accuracy']:.4f} elapsed={time.perf_counter()-started:.1f}s",flush=True)
        heartbeat("final_evaluation",cfg["epochs"])
        test=evaluate(model,data["test_x"],data["test_y"],cfg["tolerance"]) if cfg["evaluate_test"] else None
        save_weights(path/"final.npz",model,cfg,cfg["epochs"],learner)
        result=dict(status="complete",config=cfg,final_validation=metrics[-1]["validation"],
            final_test=test,final_audit=audits[-1],training_counts=counts.__dict__,
            elapsed_seconds=time.perf_counter()-started,finished_utc=utc(),
            selection="fixed final epoch; no best-checkpoint or hyperparameter selection")
        write_json(path/"result.json",result)
        heartbeat("complete",cfg["epochs"])
        return dict(case=path.name,status="complete",test_accuracy=None if test is None else test["accuracy"])
    except Exception as error:
        write_json(path/"failure.json",dict(error=repr(error),traceback=traceback.format_exc(),utc=utc()))
        heartbeat("failed",len(metrics))
        return dict(case=path.name,status="failed",error=repr(error))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--raw-dir",type=Path,default=Path("/home/filip/python2cadence/data/MNIST/raw"))
    parser.add_argument("--hidden",type=int,default=64)
    parser.add_argument("--epochs",type=int,default=5)
    parser.add_argument("--seeds",type=int,nargs="+",default=[0,1])
    parser.add_argument("--angles",type=float,nargs="+",default=[0,45,90])
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    parser.add_argument("--batch-size",type=int,default=128)
    parser.add_argument("--learning-rate",type=float,default=.001)
    parser.add_argument("--homeo-coefficient",type=float,default=1.)
    parser.add_argument("--cap",type=float,default=.95)
    parser.add_argument("--beta",type=float,default=.01)
    parser.add_argument("--noise",type=float,default=0.)
    parser.add_argument("--tolerance",type=float,default=1e-10)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--train-limit",type=int)
    parser.add_argument("--val-limit",type=int)
    parser.add_argument("--no-test",action="store_true")
    args=parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Expected a fresh output directory; got {args.output}")
    if not 0<args.cap<1 or args.epochs<1:
        raise ValueError(f"Expected 0 < cap < 1 and positive epochs; got {args.cap}, {args.epochs}")
    args.output.mkdir(parents=True)
    cache=args.output/"data"
    cache.mkdir()
    data,manifest=load_mnist(args.raw_dir,train_limit=args.train_limit,val_limit=args.val_limit)
    for name,value in data.items():
        np.save(cache/(name+".npy"),value)
    del data
    write_json(args.output/"dataset.json",manifest)
    common=dict(hidden=args.hidden,epochs=args.epochs,batch_size=args.batch_size,
        learning_rate=args.learning_rate,homeo_coefficient=args.homeo_coefficient,cap=args.cap,
        beta=args.beta,noise=args.noise,tolerance=args.tolerance,evaluate_test=not args.no_test)
    cases=[dict(common,method=method,alpha=angle,seed=seed)
           for seed in args.seeds for angle in args.angles for method in args.methods]
    case_name=lambda c:f"alpha{c['alpha']:g}_seed{c['seed']}_{c['method']}"
    git=lambda *a:subprocess.check_output(["git",*a],text=True).strip()
    source_files=[Path("labs/directed_eqprop.py"),Path(__file__),Path("labs/mnist_eqprop_data.py")]
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    run=dict(tier="exploratory",started_utc=utc(),host=platform.node(),pid=os.getpid(),
        git_head=git("rev-parse","HEAD"),git_status=git("status","--short"),source_sha256=source_hashes,
        python=platform.python_version(),torch=torch.__version__,device="cpu",threads_per_worker=1,
        workers=args.workers,cases={case_name(c):c for c in cases},
        monitoring="status heartbeat every 20 seconds; per-epoch metrics and checkpoint; terminal result.json",
        selection="No test-based selection; fixed final epoch; all declared cells retained",
        recovery="Retry operational failures only; do not modify settings to rescue poor accuracy")
    write_json(args.output/"run.json",run)
    completed=[]
    write_json(args.output/"status.json",dict(status="running",expected=len(cases),completed=0,pid=os.getpid(),heartbeat_utc=utc()))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures=[pool.submit(run_case,cfg,str(args.output/case_name(cfg)),str(cache)) for cfg in cases]
        for future in as_completed(futures):
            completed.append(future.result())
            write_json(args.output/"completed.json",completed)
            write_json(args.output/"status.json",dict(status="running",expected=len(cases),completed=len(completed),pid=os.getpid(),heartbeat_utc=utc()))
    failures=[r for r in completed if r["status"]!="complete"]
    write_json(args.output/"status.json",dict(status="failed" if failures else "complete",expected=len(cases),completed=len(completed),failed=failures,heartbeat_utc=utc()))
    if failures:
        raise SystemExit(f"{len(failures)} of {len(cases)} trajectories failed; see failure.json")


if __name__=="__main__":
    main()
