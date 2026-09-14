"""Shortened Figure 4a-d replication, including measured zero-order feedback."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from labs.fashion_eqprop_data import load_fashion_mnist
from labs.fmnist_homeostasis import METHODS, PaperNet, MeasuredFeedback, Work, free_state, diagnostics, training_gradient
from labs.tools.train_directed_eqprop_mnist import utc, write_json


@torch.no_grad()
def evaluate(model, x, y, steps, batch_size=500):
    total, correct, max_residual = 0., 0, 0.
    for start in range(0, len(x), batch_size):
        xb = torch.as_tensor(np.array(x[start:start+batch_size]), device=model.bias.device)
        yb = torch.as_tensor(np.array(y[start:start+batch_size]), device=model.bias.device)
        u, drive = free_state(model, xb, steps)
        losses, logits, _, _ = model.cost(u, yb)
        total += float(losses.sum())
        correct += int((logits.argmax(-1)==yb).sum())
        max_residual = max(max_residual, float(model.force(u, drive).norm(dim=-1).max()))
    return dict(accuracy=correct/len(x), loss=total/len(x), examples=len(x), free_residual_max=max_residual)


def save_weights(path, model, cfg, epoch, learner):
    extra = {} if learner is None else dict(feedback_matrix=learner.matrix,
                                            feedback_observations=learner.observations)
    np.savez(path, **{k:v.detach().cpu().numpy() for k,v in model.state_dict().items()},
             epoch=epoch, config_json=json.dumps(cfg), **extra)


def run_case(cfg, path, cache):
    torch.set_num_threads(1)
    path, cache = Path(path), Path(cache)
    path.mkdir()
    started = time.perf_counter()
    work, metrics, audits = Work(), [], []
    last_heartbeat = 0.
    write_json(path/"config.json",cfg)

    def heartbeat(state, epoch=0, batch=0):
        nonlocal last_heartbeat
        last_heartbeat=time.perf_counter()
        write_json(path/"status.json",dict(status=state, epoch=epoch, batch=batch,
            pid=os.getpid(), heartbeat_utc=utc(), elapsed_seconds=last_heartbeat-started,
            cpu_seconds=time.process_time(), peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            training_work=asdict(work)))

    try:
        heartbeat("loading")
        data={k:np.load(cache/(k+".npy"), mmap_mode="r") for k in ("train_x","train_y","test_x","test_y")}
        dtype=getattr(torch,cfg["dtype"])
        model=PaperNet(hidden=cfg["hidden"],seed=cfg["seed"],alpha=90.,dtype=dtype).to(cfg["device"])
        learner=MeasuredFeedback(model.n,10) if cfg["method"]=="zo_learned4" else None
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg["learning_rate"],weight_decay=0.)
        order_rng=np.random.default_rng(41000+cfg["seed"])
        probe_rng=torch.Generator(device=cfg["device"]).manual_seed(51000+cfg["seed"])
        homeo_rng=torch.Generator(device=cfg["device"]).manual_seed(61000+cfg["seed"])
        feedback_cfg=dict(learner=learner, beta=cfg["beta"],free_steps=cfg["free_steps"],response_steps=cfg["response_steps"])
        cohort_x=torch.tensor(data["train_x"][:8])
        cohort_y=torch.tensor(data["train_y"][:8])

        def audit(epoch):
            return dict(epoch=epoch,**diagnostics(model,cohort_x,cohort_y,cfg["method"],**feedback_cfg))

        save_weights(path/"initial.npz",model,cfg,0,learner)
        audits.append(audit(0))
        write_json(path/"audits.json",audits)
        for epoch in range(1,cfg["epochs"]+1):
            epoch_start=time.perf_counter()
            sums, seen = {}, 0
            order=order_rng.permutation(len(data["train_y"]))
            for batch,start in enumerate(range(0,len(order),cfg["batch_size"]),1):
                indices=order[start:start+cfg["batch_size"]]
                x=torch.as_tensor(data["train_x"][indices], device=cfg["device"])
                y=torch.as_tensor(data["train_y"][indices], device=cfg["device"])
                grads, new_work, stats=training_gradient(model,x,y,cfg["method"],probe_rng,
                    homeo_generator=homeo_rng, **feedback_cfg)
                if not all(bool(torch.isfinite(g).all()) for g in grads.values()):
                    raise RuntimeError("Non-finite training gradient; no replacement by zeros")
                optimizer.zero_grad(set_to_none=True)
                for name,p in model.named_parameters():
                    p.grad=grads[name]
                optimizer.step()
                work.add(new_work)
                seen+=len(x)
                for k,v in stats.items():
                    sums[k]=max(sums.get(k,0),v) if k.endswith("_max") else sums.get(k,0)+v*len(x)
                if time.perf_counter()-last_heartbeat>20:
                    heartbeat("training",epoch,batch)
            train={k:v if k.endswith("_max") else v/seen for k,v in sums.items()}
            heartbeat("evaluation",epoch)
            eval_x, eval_y=(data["train_x"][:200],data["train_y"][:200]) if cfg["no_test"] else (data["test_x"],data["test_y"])
            evaluated=evaluate(model,eval_x,eval_y,cfg["free_steps"])
            record=dict(epoch=epoch, train=train, evaluation=evaluated,
                        training_work=asdict(work),epoch_seconds=time.perf_counter()-epoch_start,
                        elapsed_seconds=time.perf_counter()-started)
            metrics.append(record)
            write_json(path/"metrics.json",metrics)
            if epoch in (1,5,cfg["epochs"]):
                audits.append(audit(epoch))
                write_json(path/"audits.json",audits)
            save_weights(path/"latest.npz",model,cfg,epoch,learner)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=cfg,
                epoch=epoch,work=asdict(work),metrics=metrics,audits=audits,
                probe_rng=probe_rng.get_state(),homeo_rng=homeo_rng.get_state(),order_rng=order_rng.bit_generator.state,
                learner=None if learner is None else learner.__dict__),path/"latest.pt")
            heartbeat("training",epoch)
            message=f"{path.name} epoch={epoch} eval={evaluated['accuracy']:.4f} residual={evaluated['free_residual_max']:.2g} elapsed={time.perf_counter()-started:.1f}s"
            with (path/"progress.log").open("a") as stream:
                stream.write(message+"\n")
            print(message,flush=True)
        save_weights(path/"final.npz",model,cfg,cfg["epochs"],learner)
        result=dict(status="complete",config=cfg,final_evaluation=metrics[-1]["evaluation"],
                    training_work=asdict(work),elapsed_seconds=time.perf_counter()-started,finished_utc=utc())
        write_json(path/"result.json",result)
        heartbeat("complete",cfg["epochs"])
        return dict(case=path.name,status="complete")
    except Exception as exc:
        write_json(path/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc(),utc=utc()))
        heartbeat("failed",len(metrics))
        return dict(case=path.name,status="failed",error=repr(exc))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--raw-dir",type=Path,default=Path("/home/filip/datasets/fashion_mnist/FashionMNIST/raw"))
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--hidden",type=int,default=256)
    parser.add_argument("--seeds",nargs="+",type=int,default=[0,1,2,3,4])
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=list(METHODS))
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--dtype",choices=("float32","float64"),default="float32")
    parser.add_argument("--train-limit",type=int)
    parser.add_argument("--no-test",action="store_true")
    args=parser.parse_args()
    if args.epochs<1 or args.hidden<1 or args.workers<1 or len(set(args.seeds))!=len(args.seeds) or len(set(args.methods))!=len(args.methods):
        raise ValueError(f"Expected positive sizes and unique methods/seeds; got {vars(args)}")
    if args.output.exists():
        raise ValueError(f"Expected fresh output directory; got {args.output}")
    if args.device=="cuda" and (not torch.cuda.is_available() or args.workers!=1):
        raise ValueError("Expected available CUDA and workers=1 for GPU execution")
    args.output.mkdir(parents=True)
    cache=args.output/"data"
    cache.mkdir()
    data,manifest=load_fashion_mnist(args.raw_dir,train_limit=args.train_limit,dtype=args.dtype)
    for k,v in data.items():
        np.save(cache/(k+".npy"),v)
    del data
    write_json(args.output/"dataset.json",manifest)
    common=dict(epochs=args.epochs,hidden=args.hidden,device=args.device,dtype=args.dtype,
        batch_size=50,learning_rate=1e-4,free_steps=150,response_steps=20,beta=.01,
        no_test=args.no_test,projection=False,initial_angle=90.,
        homeostasis_coefficient=1.,homeostasis_probes_per_example=5,feedback_learning_rate=.25)
    cases={f"seed{seed}_{method}":dict(common,seed=seed,method=method) for seed in args.seeds for method in args.methods}
    sources=["labs/fmnist_homeostasis.py","labs/fashion_eqprop_data.py","labs/directed_eqprop.py",
             "labs/mnist_eqprop_data.py","labs/tools/train_directed_eqprop_mnist.py",
             "labs/tools/train_fmnist_homeostasis.py"]
    frozen=args.output/"source"
    frozen.mkdir()
    hashes={}
    for name in sources:
        blob=Path(name).read_bytes()
        hashes[name]=hashlib.sha256(blob).hexdigest()
        dest=frozen/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(blob)
    write_json(args.output/"run.json",dict(tier="smoke" if args.train_limit else "shortened paper-protocol replication",
        cases=cases,source_sha256=hashes,git_head=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        git_status=subprocess.check_output(["git","status","--short"],text=True).strip(),
        command=sys.argv,started_utc=utc(),host=platform.node(),pid=os.getpid(),workers=args.workers,
        torch=torch.__version__,python=platform.python_version(),threads_per_worker=1,
        selection="fixed final epoch; no test-based selection or optimizer tuning",
        monitoring="20-second worker heartbeat, epoch metrics/checkpoints, terminal result.json; parent session monitored",
        recovery="Only operational failures may be retried with identical scientific settings; preserve failures"))
    completed=[]
    write_json(args.output/"status.json",dict(status="running",expected=len(cases),completed=0,pid=os.getpid(),utc=utc()))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures=[pool.submit(run_case,cfg,str(args.output/name),str(cache)) for name,cfg in cases.items()]
        for future in as_completed(futures):
            completed.append(future.result())
            write_json(args.output/"completed.json",completed)
            write_json(args.output/"status.json",dict(status="running",expected=len(cases),completed=len(completed),utc=utc()))
    failures=[r for r in completed if r["status"]!="complete"]
    write_json(args.output/"status.json",dict(status="failed" if failures else "complete",expected=len(cases),completed=len(completed),failures=failures,utc=utc()))
    if failures:
        raise SystemExit(f"{len(failures)} failed cases: {failures}")


if __name__=="__main__":
    main()
