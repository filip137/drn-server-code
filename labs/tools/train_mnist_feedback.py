"""Scale the four-loop physical-feedback comparison to full 28x28 MNIST.

Exploratory matched settings, fixed final epoch, no hyperparameter selection.
All EqProp methods measure centered cost-nudged equilibria; learned MC4 uses
the previous measured predictor plus four new physical probe pairs per image.
"""

import argparse
import copy
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
import sys
import time
import traceback

import numpy as np
import torch

from labs.adjoint_baselines import AveragedMomentum, MeasuredBaseline
from labs.loop_circulation_feedback import calibrate_loop_controller
from labs.mnist_eqprop_data import load_mnist
from labs.recurrent_eqprop import Classifier, flatten_gradient, parameter_gradient, settle
from labs.response_loop_feedback import calibrate_response_loops
from labs.tools.run_random_nudge_hopfield import cosine, write_csv, write_json
from labs.tools.test_structured_circulation import make_loop_case
from labs.torch_recurrent_eqprop import PhysicalBackend, tensor


METHODS = ("contrastive_ep", "known_skew_asymep", "dc_asymep", "noise_asymep", "learned_mc4")


def calibrate_case(data, size, seed, args, output):
    """Identical initial physical model, independently known wiring, fixed budget."""
    output.mkdir(parents=True)
    model, left, right, reference = make_loop_case(seed, size, args.loops, input_size=784)
    network = model.network()
    drive = model.drive(data["train_x"][:1])[0]
    free = settle(network, drive)
    center = free.state
    directions, readouts = np.concatenate((left.T, right.T)), np.concatenate((right.T, left.T))
    rng = np.random.default_rng(72000+seed)

    def matrix(coefficients):
        forward = (left*coefficients)@right.T
        return (forward-forward.T)/np.sqrt(2)

    def measure(coefficients):
        controller = matrix(coefficients)
        kw = dict(initial=center, skew_correction=-controller/2, center=center, tolerance=1e-9)
        plus = settle(network, drive-args.beta*directions, **kw)
        minus = settle(network, drive+args.beta*directions, **kw)
        yp = np.sum((plus.state-center)*readouts, axis=-1)
        yn = np.sum((minus.state-center)*readouts, axis=-1)
        yp += rng.normal(scale=args.read_noise, size=yp.shape)
        yn += rng.normal(scale=args.read_noise, size=yn.shape)
        responses = (yp-yn)/(2*args.beta)
        defects = np.sqrt(2)*(responses[:args.loops]-responses[args.loops:])
        return defects, dict(equilibrations=plus.equilibrations+minus.equilibrations,
            scalar_reads=4*args.loops, total_excitation_sq=4*args.loops*args.beta**2,
            relaxation_iterations=plus.iterations+minus.iterations)

    started = time.perf_counter()
    coefficients, counts, history = calibrate_response_loops(
        measure, args.loops, steps=args.dc_steps, learning_rate=.2)
    dc = matrix(coefficients)
    counts.update(free_anchor_equilibrations=1, anchor_projection_reads=2*args.loops,
                  learned_coefficients=args.loops, known_wiring_coefficients=2*size*args.loops,
                  process_noise_channels=0)
    dc_seconds = time.perf_counter()-started
    write_csv(output/"dc_calibration.csv", history)
    # The learner receives the opaque physical force, projected wiring and
    # fixed duration. Neither calibration receives reference gains or a J.
    noise = calibrate_loop_controller(lambda states: network.force(states, drive), center,
        left, right, duration=100., dt=.02, temperature=.05, replicas=8,
        learning_rate=.1, update_interval=10, burn_in=10., average_after=30.,
        seed=64000+seed, read_noise=0.)
    write_csv(output/"noise_calibration.csv", noise.history)
    np.savez_compressed(output/"controllers.npz", dc_controller=dc, noise_controller=noise.matrix,
        dc_coefficients=coefficients, noise_coefficients=noise.coefficients, left=left, right=right,
        center=center, **reference)
    np.savez_compressed(output/"initial_model.npz", symmetric=model.symmetric, skew=model.skew,
                        inputs=model.inputs, bias=model.bias)
    summaries = dict(size=size, seed=seed, calibration_training_index=int(data["train_indices"][0]),
        known_basis_exact=True, controller_frozen=True, task_probes_after_calibration=0,
        dc=dict(controller_relative_error=float(np.linalg.norm(dc+model.skew)/np.linalg.norm(model.skew)),
                counts=counts, elapsed_seconds=dc_seconds, steps=args.dc_steps,
                rate=.2, amplitude=args.beta, scalar_read_noise=args.read_noise),
        noise=dict(controller_relative_error=float(np.linalg.norm(noise.matrix+model.skew)/np.linalg.norm(model.skew)),
                   counts=noise.stats, free_anchor_equilibrations=1))
    write_json(output/"calibration_summary.json", summaries)
    return summaries


@torch.no_grad()
def evaluate(model, backend, x, y, batch_size=1000):
    network = model.network()
    loss, correct, residual, iterations = 0., 0, 0., 0
    for begin in range(0, len(x), batch_size):
        xb = x[begin:begin+batch_size]
        labels = y[begin:begin+batch_size]
        free, _, _, _ = backend.free(model, xb, network=network)
        logits = model.logit_scale*free.state[:, model.hidden:]
        loss += float(torch.nn.functional.cross_entropy(logits, labels, reduction="sum"))
        correct += int((logits.argmax(dim=-1) == labels).sum())
        residual = max(residual, free.residual)
        iterations += free.iterations
    return dict(loss=loss/len(x), accuracy=correct/len(x), correct=correct,
                examples=len(x), equilibrations=len(x), residual=residual,
                relaxation_iterations=iterations)


def diagnostic_audit(model, backend, x, y, method, learner, controller, config):
    # Read-only replay on a fixed 16-training-image cohort. Additional
    # measurements are charged separately and never train the predictor.
    gradient, meta, observed = backend.gradient(model, x, y, method, np.random.default_rng(83000),
        beta=config["beta"], sigma=config["read_noise"], controller=controller,
        learner=copy.deepcopy(learner), return_audit=True)
    jacobian = model.network().jacobian(observed["state"])
    truth = np.linalg.solve(jacobian.transpose(0, 2, 1), observed["cost"][..., None])[..., 0]
    exact = parameter_gradient(model, x.cpu().numpy(), observed["state"], truth)
    actual_flat, exact_flat = flatten_gradient(gradient), flatten_gradient(exact)
    row = dict(gradient_cosine=cosine(actual_flat, exact_flat),
        gradient_relative_error=float(np.linalg.norm(actual_flat-exact_flat)/np.linalg.norm(exact_flat)),
        adjoint_relative_error=float(np.linalg.norm(observed["feedback"]-truth)/np.linalg.norm(truth)),
        baseline_relative_error=float(np.linalg.norm(observed["baseline"]-truth)/np.linalg.norm(truth)),
        physical_audit_equilibrations=meta["equilibrations"], cohort_examples=len(x),
        oracle_access="post-estimation diagnostic only; model/predictor/optimizer unchanged")
    for name, g, ref in zip(("symmetric", "inputs", "bias"), gradient, exact):
        row[name+"_gradient_cosine"] = cosine(g, ref)
    return row


def save_checkpoint(path, model, optimizer, learner, controller, epoch, config):
    extra = {}
    if learner is not None:
        extra.update(predictor_matrix=learner.matrix, predictor_observations=learner.observations,
                     predictor_relaxation=learner.relaxation)
    if controller is not None:
        extra["feedback_controller"] = controller
    if optimizer.moments is not None:
        extra.update({"ema_"+name:part for name, part in zip(("symmetric", "inputs", "bias"), optimizer.moments)})
    temporary = path.with_name(path.stem+".tmp.npz")
    np.savez_compressed(temporary, symmetric=model.symmetric, skew=model.skew,
        inputs=model.inputs, bias=model.bias, epoch=epoch, outputs=model.outputs,
        cubic=model.cubic, logit_scale=model.logit_scale, symmetric_cap=model.symmetric_cap,
        momentum=optimizer.momentum, optimizer_steps=optimizer.steps,
        config_json=json.dumps(config), **extra)
    temporary.replace(path)


def train_worker(payload):
    config, size, seed, method, root = payload
    torch.set_num_threads(1)
    root = Path(root)
    output = root/f"n{size}_seed{seed}"/method
    output.mkdir()
    started = time.perf_counter()
    status = dict(status="running", size=size, seed=seed, method=method, pid=os.getpid(),
                  completed_epochs=0, target_epochs=config["epochs"])

    def progress(message, **fields):
        status.update(fields, detail=message, elapsed_seconds=time.perf_counter()-started,
                      updated_utc=datetime.now(timezone.utc).isoformat(),
                      user_cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write_json(output/"status.json", status)
        line = f"n={size} seed={seed} {method}: {message}"
        with (output/"run.log").open("a") as stream:
            stream.write(line+"\n")
        print(line, flush=True)

    try:
        progress("loading official MNIST and frozen calibration")
        data, _ = load_mnist(config["raw_dir"], train_limit=config["train_limit"], val_limit=config["val_limit"])
        backend = PhysicalBackend(config["device"], graphs=not config["no_graphs"])
        device = backend.device
        tx = tensor(data["train_x"], device)
        ty = torch.as_tensor(data["train_y"], device=device)
        vx = tensor(data["val_x"], device)
        vy = torch.as_tensor(data["val_y"], device=device)
        # Keep test inputs off the accelerator until the fixed final epoch.
        initial = np.load(output.parent/"initial_model.npz")
        model = Classifier(**{key:initial[key].copy() for key in ("symmetric", "skew", "inputs", "bias")})
        controllers = np.load(output.parent/"controllers.npz")
        controller = controllers[method.split("_")[0]+"_controller"].copy() if method in ("dc_asymep", "noise_asymep") else None
        learner = MeasuredBaseline.zeros(size, 10, .25) if method == "learned_mc4" else None
        optimizer = AveragedMomentum(config["momentum"])
        shuffle, rng = np.random.default_rng(10000+seed), np.random.default_rng(20000+seed)
        cumulative = dict(training_equilibrations=0, training_state_reads=0,
                          total_probe_excitation_sq=0., total_error_excitation_sq=0.,
                          training_batched_iterations=0, training_max_residual=0.,
                          evaluation_equilibrations=0, audit_equilibrations=0,
                          optimizer_steps=0, projected_updates=0)
        rows, audits = [], []
        validation = evaluate(model, backend, vx, vy, config["eval_batch_size"])
        cumulative["evaluation_equilibrations"] += validation["equilibrations"]
        rows.append(dict(epoch=0, validation_loss=validation["loss"], validation_accuracy=validation["accuracy"],
                         elapsed_seconds=time.perf_counter()-started, **cumulative))
        audit_epochs = sorted(set([0, config["epochs"]]+[e for e in (5, 10) if e < config["epochs"]]))
        audit = diagnostic_audit(model, backend, tx[:16], ty[:16], method, learner, controller, config)
        cumulative["audit_equilibrations"] += audit["physical_audit_equilibrations"]
        audits.append(dict(epoch=0, **audit))
        write_csv(output/"audits.csv", audits)
        progress(f"initial validation={validation['accuracy']:.4f}")
        last_heartbeat = time.perf_counter()
        for epoch in range(1, config["epochs"]+1):
            order = shuffle.permutation(len(tx))
            for batch_index, begin in enumerate(range(0, len(order), config["batch_size"])):
                indices = torch.as_tensor(order[begin:begin+config["batch_size"]], device=device)
                gradient, meta, _ = backend.gradient(model, tx[indices], ty[indices], method, rng,
                    beta=config["beta"], sigma=config["read_noise"], controller=controller, learner=learner)
                cumulative["projected_updates"] += int(model.update(
                    optimizer.direction(gradient), config["learning_rate"]))
                cumulative["optimizer_steps"] += 1
                cumulative["training_equilibrations"] += meta["equilibrations"]
                cumulative["training_state_reads"] += meta["state_reads"]
                cumulative["training_batched_iterations"] += meta["relaxation_iterations"]
                cumulative["training_max_residual"] = max(cumulative["training_max_residual"], meta["max_residual"])
                for key in ("total_probe_excitation_sq", "total_error_excitation_sq"):
                    cumulative[key] += meta[key]
                if time.perf_counter()-last_heartbeat >= 20:
                    progress(f"epoch {epoch}/{config['epochs']}, batch {batch_index+1}/{int(np.ceil(len(tx)/config['batch_size']))}",
                             active_epoch=epoch, active_batch=batch_index+1, **cumulative)
                    last_heartbeat = time.perf_counter()
            validation = evaluate(model, backend, vx, vy, config["eval_batch_size"])
            cumulative["evaluation_equilibrations"] += validation["equilibrations"]
            if epoch in audit_epochs:
                audit = diagnostic_audit(model, backend, tx[:16], ty[:16], method, learner, controller, config)
                cumulative["audit_equilibrations"] += audit["physical_audit_equilibrations"]
                audits.append(dict(epoch=epoch, **audit))
                write_csv(output/"audits.csv", audits)
            rows.append(dict(epoch=epoch, validation_loss=validation["loss"], validation_accuracy=validation["accuracy"],
                validation_force_residual=validation["residual"], elapsed_seconds=time.perf_counter()-started,
                predictor_observations=0 if learner is None else learner.observations, **cumulative))
            write_csv(output/"metrics.csv", rows)
            save_checkpoint(output/"latest.npz", model, optimizer, learner, controller, epoch, config)
            progress(f"completed epoch {epoch}/{config['epochs']}, validation={validation['accuracy']:.4f}",
                     completed_epochs=epoch, **cumulative)
        test = None
        if not config["no_test"]:
            test = evaluate(model, backend, tensor(data["test_x"], device),
                            torch.as_tensor(data["test_y"], device=device), config["eval_batch_size"])
            cumulative["evaluation_equilibrations"] += test["equilibrations"]
        expected = (9 if method == "learned_mc4" else 3)*len(tx)*config["epochs"]
        if cumulative["training_equilibrations"] != expected:
            raise RuntimeError(f"Expected {expected} training equilibrations; got {cumulative['training_equilibrations']}")
        result = dict(status="complete", size=size, seed=seed, method=method,
            epochs=config["epochs"], train_examples=len(tx), validation=validation, test=test,
            final_epoch_selected_in_advance=True, hyperparameter_search=False,
            training_rule="measured-baseline MC4 plus local parameter-force gradient" if method == "learned_mc4" else "centered contrastive EqProp",
            probe_pairs_per_training_example=4 if method == "learned_mc4" else 0,
            calibration_shared_from_case=True, final_audit=audits[-1], initial_audit=audits[0],
            elapsed_seconds=time.perf_counter()-started, **cumulative)
        save_checkpoint(output/"final.npz", model, optimizer, learner, controller, config["epochs"], config)
        write_json(output/"result.json", result)
        progress(f"complete; test={None if test is None else test['accuracy']}", status="complete")
        return result
    except BaseException as error:
        (output/"traceback.txt").write_text(traceback.format_exc())
        progress(f"failed: {type(error).__name__}: {error}", status="failed")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--eval-batch-size", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=.1)
    parser.add_argument("--momentum", type=float, default=.9)
    parser.add_argument("--beta", type=float, default=.01)
    parser.add_argument("--read-noise", type=float, default=1e-5)
    parser.add_argument("--loops", type=int, default=4)
    parser.add_argument("--dc-steps", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--no-test", action="store_true", help="Operational smoke: never evaluate test metrics")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Expected a fresh output directory; got {args.output}")
    for name in ("epochs", "batch_size", "eval_batch_size", "loops", "dc_steps", "workers"):
        if getattr(args, name) < 1:
            parser.error(f"Expected positive {name}; got {getattr(args, name)}")
    if args.device.startswith("cuda") and args.workers != 1:
        parser.error(f"Expected one worker for CUDA; got {args.workers}")
    if not 0 <= args.momentum < 1 or args.learning_rate <= 0 or args.beta <= 0 or args.read_noise < 0:
        parser.error("Expected positive LR/beta, nonnegative noise and momentum in [0,1)")
    for values in (args.sizes, args.seeds, args.methods):
        if len(set(values)) != len(values):
            parser.error(f"Expected distinct coverage values; got {values}")
    if min(args.sizes) < 10+args.loops or min(args.seeds) < 0 or args.loops > 10:
        parser.error("Expected nonnegative seeds and 1 <= loops <= min(10, hidden)")
    args.output.mkdir(parents=True)
    config = {k:str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__), root/"labs/torch_recurrent_eqprop.py", root/"labs/mnist_eqprop_data.py"]
    source_hashes = {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    run = dict(config=config, evidence_tier="exploratory matched comparison, full MNIST",
        command=[sys.executable, "-m", "labs.tools.train_mnist_feedback", *sys.argv[1:]],
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        source_status=subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        source_hashes=source_hashes, python=sys.version, numpy=np.__version__, torch=torch.__version__,
        platform=platform.platform(), hostname=platform.node(), pid=os.getpid(), dtype="float64",
        force_tolerance=1e-9, expected_calibrations=2*len(args.sizes)*len(args.seeds),
        expected_training=len(args.sizes)*len(args.seeds)*len(args.methods),
        known_structure=f"{args.loops} fixed skew gains in a known independently wired basis; no arbitrary dense identification",
        evaluation="validation each epoch; official test only at the predeclared final epoch",
        optimizer="bias-corrected gradient EMA then spectral-projected SGD; MC predictor is separate")
    write_json(args.output/"run.json", run)
    write_json(args.output/"config.json", config)
    torch.set_num_threads(1)
    started = time.perf_counter()
    state = dict(status="calibrating", completed_calibrations=0, completed_training=0, failed_training=0)
    write_json(args.output/"status.json", state)
    try:
        data, manifest = load_mnist(args.raw_dir, train_limit=args.train_limit, val_limit=args.val_limit)
        write_json(args.output/"dataset.json", manifest)
        np.savez_compressed(args.output/"dataset_split.npz", **{key:data[key] for key in
            ("train_indices", "val_indices", "test_indices", "feature_mean", "feature_std")})
        calibration = []
        for size in args.sizes:
            for seed in args.seeds:
                row = calibrate_case(data, size, seed, args, args.output/f"n{size}_seed{seed}")
                calibration.append(row)
                state["completed_calibrations"] += 2
                state["elapsed_seconds"] = time.perf_counter()-started
                write_json(args.output/"calibration.json", calibration)
                write_json(args.output/"status.json", state)
                print(f"n={size} seed={seed} calibrated: DC error={row['dc']['controller_relative_error']:.4f}, noise error={row['noise']['controller_relative_error']:.4f}", flush=True)
        del data
        scheduling_order = sorted(args.methods, key=lambda method: method != "learned_mc4")
        payloads = [(config, n, seed, method, str(args.output.resolve()))
                    for n in sorted(args.sizes, reverse=True) for method in scheduling_order for seed in args.seeds]
        # Submit larger jobs first, with bounded single-BLAS-thread workers.
        results, failures = [], []
        state["status"] = "training"
        write_json(args.output/"status.json", state)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(train_worker, payload):payload[1:4] for payload in payloads}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    results.append(future.result())
                    state["completed_training"] += 1
                except BaseException as error:
                    failures.append(dict(case=list(case), error=repr(error)))
                    state["failed_training"] += 1
                state["elapsed_seconds"] = time.perf_counter()-started
                write_json(args.output/"results.json", results)
                write_json(args.output/"failures.json", failures)
                write_json(args.output/"status.json", state)
        state["status"] = "failed" if failures else "complete"
        write_json(args.output/"status.json", state)
        if failures:
            raise RuntimeError(f"Expected all {len(payloads)} trajectories complete; got {len(failures)} failures")
        if len(results) != run["expected_training"]:
            raise RuntimeError(f"Expected {run['expected_training']} results; got {len(results)}")
    except BaseException:
        state["status"] = "failed"
        (args.output/"traceback.txt").write_text(traceback.format_exc())
        write_json(args.output/"status.json", state)
        raise


if __name__ == "__main__":
    main()
