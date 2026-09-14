"""Check physical GPU/CPU solver parity and estimate full-MNIST training cost."""

import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch

from labs.adjoint_baselines import AveragedMomentum, MeasuredBaseline
from labs.mnist_eqprop_data import load_mnist
from labs.recurrent_eqprop import flatten_gradient
from labs.tools.run_random_nudge_hopfield import write_json
from labs.tools.test_structured_circulation import make_loop_case
from labs.tools.train_recurrent_eqprop_digits import gradient_step
from labs.torch_recurrent_eqprop import PhysicalBackend, tensor


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", default=250, type=int)
    p.add_argument("--sizes", nargs="+", type=int, default=[64, 256])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    data, manifest = load_mnist(args.raw_dir, train_limit=1000, val_limit=1000)
    results = []
    backend = PhysicalBackend(args.device)
    for n in args.sizes:
        initial, _, _, _ = make_loop_case(0, n, 4, input_size=784)
        for method in ("contrastive_ep", "known_skew_asymep", "dc_asymep", "learned_mc4"):
            model_cpu, model_device = copy.deepcopy(initial), copy.deepcopy(initial)
            controller = -.98*initial.skew if method == "dc_asymep" else None
            cpu_method = "circulation_asymep" if controller is not None else method
            learners = [MeasuredBaseline.zeros(n, 10) if method == "learned_mc4" else None for _ in range(2)]
            rngs = [np.random.default_rng(22), np.random.default_rng(22)]
            optimizers = [AveragedMomentum(.9), AveragedMomentum(.9)]
            max_error = 0.
            for step in range(3):
                x, y = data["train_x"][step*8:(step+1)*8], data["train_y"][step*8:(step+1)*8]
                reference, rm = gradient_step(model_cpu, x, y, cpu_method, rngs[0], beta=.01,
                    sigma=1e-5, learner=learners[0], feedback_controller=controller)
                actual, am, _ = backend.gradient(model_device, x, y, method, rngs[1], beta=.01,
                    sigma=1e-5, learner=learners[1], controller=controller)
                error = np.linalg.norm(flatten_gradient(actual)-flatten_gradient(reference))
                error /= max(np.linalg.norm(flatten_gradient(reference)), 1e-20)
                assert error < 1e-5, (n, method, step, error)
                assert am["equilibrations"] == rm["equilibrations"]
                assert am["max_residual"] <= 1e-9
                for mod, opt, grad in zip((model_cpu, model_device), optimizers, (reference, actual)):
                    mod.update(opt.direction(grad), .1)
                max_error = max(max_error, error)
            parameter_error = np.linalg.norm(flatten_gradient((model_cpu.symmetric, model_cpu.inputs, model_cpu.bias))-
                flatten_gradient((model_device.symmetric, model_device.inputs, model_device.bias)))
            if method == "learned_mc4":
                np.testing.assert_allclose(learners[0].matrix, learners[1].matrix, atol=1e-5, rtol=1e-4)
            model = copy.deepcopy(initial)
            learner = MeasuredBaseline.zeros(n, 10) if method == "learned_mc4" else None
            x = tensor(data["train_x"][:args.batch_size], args.device)
            y = torch.as_tensor(data["train_y"][:args.batch_size], device=args.device)
            optimizer = AveragedMomentum(.9)
            times, iteration_counts = [], []
            for repeat in range(6):
                start = time.perf_counter()
                g, meta, _ = backend.gradient(model, x, y, method, np.random.default_rng(repeat),
                                             learner=learner, controller=controller)
                model.update(optimizer.direction(g), .1)
                if repeat:
                    times.append(time.perf_counter()-start)
                    iteration_counts.append(meta["relaxation_iterations"])
            row = dict(size=n, method=method, three_step_max_gradient_relative_error=max_error,
                three_step_parameter_l2_difference=float(parameter_error),
                median_batch_seconds=float(np.median(times)), batch_size=args.batch_size,
                estimated_15_epoch_training_seconds=float(np.median(times)*np.ceil(55000/args.batch_size)*15),
                median_batched_iterations=float(np.median(iteration_counts)),
                max_force_residual=meta["max_residual"])
            print(row, flush=True)
            results.append(row)
            write_json(args.output/"results.json", results)
    write_json(args.output/"summary.json", dict(status="passed", dtype="float64", tolerance=1e-9,
        device=args.device, torch=torch.__version__, dataset=manifest, results=results))


if __name__ == "__main__":
    main()
