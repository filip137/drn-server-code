"""Double-precision accelerator for the existing physical Hopfield experiments.

Forward relaxation and measured EqProp/probe updates only; no autograd. CUDA
graphs amortize launch overhead over short blocks of the same IMEX iterations.
Every returned equilibrium is checked against the actual force, in float64.
The NumPy model retains the identical spectral projection and optimizer.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from labs.recurrent_eqprop import Classifier


def tensor(array, device):
    return torch.as_tensor(np.array(array, copy=True), dtype=torch.float64, device=device)


def cost_gradient(state, labels, hidden, scale=4.0):
    gradient = torch.zeros_like(state)
    error = torch.softmax(scale * state[:, hidden:], dim=-1)
    error = error - torch.nn.functional.one_hot(labels, state.shape[-1]-hidden)
    gradient[:, hidden:] = scale * error
    return gradient


@dataclass
class Settled:
    state: torch.Tensor
    iterations: int
    residual: float
    equilibrations: int


class Relaxer:
    """Reusable solver for one shape; all captured inputs live in fixed buffers."""

    def __init__(self, batch, size, *, device="cpu", cubic=.25, hidden=None,
                 logit_scale=4.0, block=10, graphs=True):
        self.device = torch.device(device)
        self.cubic, self.hidden, self.logit_scale = cubic, hidden, logit_scale
        self.block = block
        self.state = torch.zeros((batch, size), dtype=torch.float64, device=device)
        self.drive = torch.zeros_like(self.state)
        self.weights = torch.zeros((size, size), dtype=torch.float64, device=device)
        self.labels = torch.zeros(batch, dtype=torch.int64, device=device)
        self.cost_scale = torch.zeros((batch, 1), dtype=torch.float64, device=device)
        self.dt = torch.full((), .2, dtype=torch.float64, device=device)
        self.residual = torch.zeros((), dtype=torch.float64, device=device)
        self.graph = None
        if graphs and self.device.type == "cuda":
            # Initialize outside capture, warm up on a side stream, then reuse.
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._block()
            torch.cuda.current_stream(device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._block()

    def coupled(self):
        value = self.state @ self.weights.T + self.drive
        if self.hidden is not None:
            value = value-self.cost_scale*cost_gradient(
                self.state, self.labels, self.hidden, self.logit_scale)
        return value

    def _block(self):
        a = 1+self.dt
        if self.cubic:
            k = torch.sqrt(3*self.dt*self.cubic/a)
        for _ in range(self.block):
            value = self.state+self.dt*self.coupled()
            if self.cubic:
                self.state.copy_(2/k*torch.sinh(torch.asinh(1.5*k*value/a)/3))
            else:
                self.state.copy_(value/a)
        force = self.coupled()-self.state-self.cubic*self.state**3
        self.residual.copy_(torch.max(torch.abs(force)))

    @torch.no_grad()
    def solve(self, weights, drive, *, initial=None, margin, weight_norm,
              labels=None, cost_scale=None, tolerance=1e-9, max_steps=10000):
        if tolerance <= 0 or max_steps < self.block:
            raise ValueError(f"Expected positive tolerance and max_steps >= block; got {tolerance}, {max_steps}")
        self.weights.copy_(weights)
        self.drive.copy_(drive)
        self.state.zero_() if initial is None else self.state.copy_(initial)
        self.cost_scale.zero_()
        bound = 0.0
        if cost_scale is not None:
            if labels is None or self.hidden is None:
                raise ValueError("Expected labels and cost-enabled solver for cost nudges")
            self.labels.copy_(labels)
            self.cost_scale.copy_(cost_scale.reshape(-1, 1))
            bound = float(cost_scale.abs().max())*self.logit_scale**2/2
        if margin <= bound:
            raise ValueError(f"Expected cost bound below stability margin; got {bound} >= {margin}")
        self.dt.fill_(min(.8, (margin-bound)/max((weight_norm+bound)**2-1, 1e-12)))
        for iterations in range(self.block, max_steps+1, self.block):
            if self.graph is not None:
                self.graph.replay()
            else:
                self._block()
            residual = float(self.residual)
            if not math.isfinite(residual):
                raise RuntimeError(f"Expected finite force residual; got {residual} at iteration {iterations}")
            if residual <= tolerance:
                return Settled(self.state.clone(), iterations, residual, len(drive))
        raise RuntimeError(f"Expected force residual <= {tolerance}; got {residual} after {max_steps} iterations")


class PhysicalBackend:
    """Batch states on device; small model/projection and learned predictor on CPU.

    The feedback controller is calibrated externally and held fixed. Access to
    the physical coupling matrix here implements the simulator, not a learned
    Jacobian or adjoint. Exact references belong in a separate audit function.
    """

    def __init__(self, device="cpu", *, graphs=True, block=10, tolerance=1e-9):
        self.device = torch.device(device)
        self.graphs, self.block, self.tolerance = graphs, block, tolerance
        self.solvers = {}

    def solve(self, model, weights, drive, *, margin, weight_norm, initial=None,
              labels=None, cost_scale=None):
        key = (len(drive), model.size, model.cubic,
               model.hidden if cost_scale is not None else None, model.logit_scale)
        if key not in self.solvers:
            self.solvers[key] = Relaxer(len(drive), model.size, device=self.device,
                cubic=model.cubic, hidden=key[3], logit_scale=model.logit_scale,
                block=self.block, graphs=self.graphs)
        return self.solvers[key].solve(weights, drive, initial=initial, margin=margin,
            weight_norm=weight_norm, labels=labels, cost_scale=cost_scale,
            tolerance=self.tolerance)

    def drive(self, model, x):
        drive = tensor(model.bias, self.device).expand(len(x), -1).clone()
        drive[:, :model.hidden] += x @ tensor(model.inputs, self.device).T
        return drive

    @torch.no_grad()
    def free(self, model, x, *, network=None):
        network = model.network() if network is None else network
        weights = tensor(network.weights, self.device)
        drive = self.drive(model, x)
        settled = self.solve(model, weights, drive, margin=network.margin,
                             weight_norm=network.weight_norm)
        return settled, drive, weights, network

    @torch.no_grad()
    def gradient(self, model, x, labels, method, rng, *, beta=.01, sigma=1e-5,
                 controller=None, learner=None, return_audit=False):
        if method not in ("contrastive_ep", "known_skew_asymep", "dc_asymep",
                          "noise_asymep", "learned_mc4"):
            raise ValueError(f"Expected a supported physical training method; got {method!r}")
        x = x if isinstance(x, torch.Tensor) else tensor(x, self.device)
        labels = torch.as_tensor(labels, dtype=torch.int64, device=self.device)
        free, drive, weights, network = self.free(model, x)
        c = cost_gradient(free.state, labels, model.hidden, model.logit_scale)
        meta = dict(equilibrations=len(x), state_reads=len(x),
            relaxation_iterations=free.iterations, max_residual=free.residual,
            probe_count=0, total_probe_excitation_sq=0., total_error_excitation_sq=0.)
        if method == "learned_mc4":
            if learner is None:
                raise ValueError("Expected a persistent measured baseline for learned_mc4")
            state_np, c_np = free.state.cpu().numpy(), c.cpu().numpy()
            baseline = tensor(learner.predict(state_np, c_np, model.cubic), self.device)
            # Identical per-example independent, unit-norm Rademacher design.
            probes_np = (2*rng.integers(0, 2, size=(len(x), 4, model.size))-1)/math.sqrt(model.size)
            probes = tensor(probes_np, self.device)
            d = drive[:, None, :].expand(-1, 4, -1).reshape(-1, model.size)
            u = probes.reshape(-1, model.size)
            center = free.state[:, None, :].expand(-1, 4, -1).reshape(-1, model.size)
            perturbed = self.solve(model, weights, torch.cat((d-beta*u, d+beta*u)),
                initial=torch.cat((center, center)), margin=network.margin,
                weight_norm=network.weight_norm)
            plus, minus = perturbed.state.chunk(2)
            response = ((plus-minus)/(2*beta)).reshape(len(x), 4, model.size)
            if sigma:
                response += tensor(rng.normal(scale=sigma/(math.sqrt(2)*beta),
                                               size=response.shape), self.device)
            readings = torch.einsum("bmn,bn->bm", response, c)
            residual = readings-torch.einsum("bmn,bn->bm", probes, baseline)
            feedback = baseline+(model.size/4)*torch.einsum("bmn,bm->bn", probes, residual)
            recurrent = -feedback.T @ free.state/len(x)
            recurrent = (recurrent+recurrent.T)/2
            inputs = -feedback[:, :model.hidden].T @ x/len(x)
            bias = -feedback.mean(dim=0)
            meta["predictor_previous_observations"] = learner.observations
            # All present predictions/gradients are formed before this CPU
            # sequential rank-one update, retaining the original algorithm.
            learner.observe(state_np, c_np, model.cubic, probes_np, readings.cpu().numpy())
            meta.update(predictor_observations=learner.observations, probe_count=4,
                        total_probe_excitation_sq=8*len(x)*beta**2)
        else:
            if method in ("dc_asymep", "noise_asymep") and controller is None:
                raise ValueError("Expected a frozen calibrated controller")
            correction = (-model.skew if method == "known_skew_asymep" else controller)
            w, d = weights, drive
            norm = network.weight_norm
            if correction is not None:
                # Training force is F(s)+2*C*(s-s0)-beta*c(s).
                w = tensor(network.weights+2*correction, self.device)
                d = drive-2*free.state @ tensor(correction, self.device).T
                norm = float(np.linalg.norm(network.weights+2*correction, 2))
            effective = beta/torch.maximum(torch.linalg.vector_norm(c, dim=-1),
                                          torch.ones(len(x), device=self.device))
            perturbed = self.solve(model, w, torch.cat((d, d)),
                initial=torch.cat((free.state, free.state)), margin=network.margin,
                weight_norm=norm, labels=torch.cat((labels, labels)),
                cost_scale=torch.cat((effective, -effective)))
            plus, minus = perturbed.state.chunk(2)
            if sigma:
                plus = plus+tensor(rng.normal(scale=sigma, size=plus.shape), self.device)
                minus = minus+tensor(rng.normal(scale=sigma, size=minus.shape), self.device)
            feedback = (plus-minus)/(2*effective[:, None])
            baseline = feedback
            coefficient = 1/(4*effective)
            recurrent = -((plus*coefficient[:, None]).T@plus
                          -(minus*coefficient[:, None]).T@minus)/len(x)
            inputs = -feedback[:, :model.hidden].T @ x/len(x)
            bias = -feedback.mean(dim=0)
            meta["total_error_excitation_sq"] = float(2*torch.sum(
                (effective*torch.linalg.vector_norm(c, dim=-1))**2))
        recurrent.fill_diagonal_(0)
        meta["equilibrations"] += perturbed.equilibrations
        meta["state_reads"] += perturbed.equilibrations
        # Block iterations are batched simulator steps, not physical time.
        meta["relaxation_iterations"] += perturbed.iterations
        meta["max_residual"] = max(meta["max_residual"], perturbed.residual)
        gradient = tuple(t.cpu().numpy() for t in (recurrent, inputs, bias))
        audit = None
        if return_audit:
            audit = dict(state=free.state.cpu().numpy(), cost=c.cpu().numpy(),
                         feedback=feedback.cpu().numpy(), baseline=baseline.cpu().numpy())
        return gradient, meta, audit
