"""Figure 4a-d homeostasis model and measured feedback, with a fixed step budget.

Reimplementation of the equations and public configuration of Laborieux & Zenke:
https://github.com/Laborieux-Axel/generalized-holo-ep/tree/30592f576bd4d4a20d3c13632f0792b0fa452781
The authors use +beta*c; here the F-b=0 convention reverses the measured
response sign and the local gradient uses -partial_theta(F)^T lambda.
See docs/fmnist_homeostasis_replication.md for all departures from that source.
"""

from dataclasses import dataclass, asdict
import math

import numpy as np
import torch

from labs.directed_eqprop import DirectedNet, LearnedFeedback, activation, slope, local_gradient


METHODS = ("vf_ad", "vf_ad_homeo", "vf_central", "zo_learned4")


@torch.jit.script
def recur(r, f1: torch.Tensor, f2: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor):
    h = f1.shape[0]
    return torch.cat((r[:, h:2*h]@b1.T,
                      r[:, :h]@f1.T+r[:, 2*h:]@b2.T,
                      r[:, h:2*h]@f2.T), dim=-1)


@torch.jit.script
def advance(u: torch.Tensor, drive: torch.Tensor, current: torch.Tensor,
            f1: torch.Tensor, f2: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor,
            steps: int):
    for _ in range(steps):
        u = recur(torch.sigmoid(4*u-2), f1, f2, b1, b2)+drive-current
    return u


@torch.jit.script
def advance_error(u: torch.Tensor, drive: torch.Tensor, target: torch.Tensor,
                  f1: torch.Tensor, f2: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor,
                  readout: torch.Tensor, readout_bias: torch.Tensor, beta: float, steps: int):
    o = readout.shape[0]
    for _ in range(steps):
        r = torch.sigmoid(4*u-2)
        c = ((r[:, -o:]@readout.T+readout_bias).softmax(-1)-target)@readout
        new = recur(r, f1, f2, b1, b2)+drive
        new[:, -o:] -= beta*c
        u = new
    return u


@torch.jit.script
def forward_tangent(u: torch.Tensor, drive: torch.Tensor, target: torch.Tensor,
                    f1: torch.Tensor, f2: torch.Tensor, b1: torch.Tensor, b2: torch.Tensor,
                    readout: torch.Tensor, readout_bias: torch.Tensor, steps: int):
    """Exact forward derivative of the finite zero-beta trajectory, as N=0 in JAX."""
    v = torch.zeros_like(u)
    o = readout.shape[0]
    for _ in range(steps):
        r = torch.sigmoid(4*u-2)
        c = ((r[:, -o:]@readout.T+readout_bias).softmax(-1)-target)@readout
        v = recur(4*r*(1-r)*v, f1, f2, b1, b2)
        v[:, -o:] -= c
        u = recur(r, f1, f2, b1, b2)+drive
    r = torch.sigmoid(4*u-2)
    return 4*r*(1-r)*v


class PaperNet(DirectedNet):
    """784-256-256-10 + 10x10 readout; no recurrent projection or centering.

    Flax Dense's default LeCun truncated normal distribution is reproduced;
    Torch and JAX random streams differ. Reverse layers use their own fan-in.
    """

    def __init__(self, inputs=784, hidden=256, outputs=10, seed=0, alpha=90., dtype=torch.float32):
        torch.nn.Module.__init__(self)
        self.inputs, self.hidden, self.outputs = inputs, hidden, outputs
        self.sizes, self.n = (hidden, hidden, outputs), 2*hidden+outputs
        gen = torch.Generator().manual_seed(seed)

        def dense(post, pre):
            # Uniform inverse CDF sampling of N(0,1) truncated at +/-2.
            p = .02275013194817921 + .9544997361036416*torch.rand(post, pre, generator=gen, dtype=dtype)
            return math.sqrt(2)*torch.erfinv(2*p-1)/(.8796256610342398*math.sqrt(pre))

        self.input = torch.nn.Parameter(dense(hidden, inputs))
        self.forward1 = torch.nn.Parameter(dense(hidden, hidden))
        self.forward2 = torch.nn.Parameter(dense(outputs, hidden))
        a = math.radians(alpha)
        self.backward1 = torch.nn.Parameter(math.cos(a)*self.forward1.detach().T+math.sin(a)*dense(hidden, hidden))
        self.backward2 = torch.nn.Parameter(math.cos(a)*self.forward2.detach().T+math.sin(a)*dense(hidden, outputs))
        self.bias = torch.nn.Parameter(torch.zeros(self.n, dtype=dtype))
        self.readout = torch.nn.Parameter(dense(outputs, outputs))
        self.readout_bias = torch.nn.Parameter(torch.zeros(outputs, dtype=dtype))

    def weights(self):
        return self.forward1, self.forward2, self.backward1, self.backward2

    def recurrent(self, r):
        return recur(r, *self.weights())

    def dense_w(self):
        eye = torch.eye(self.n, dtype=self.bias.dtype, device=self.bias.device)
        return self.recurrent(eye).T

    def project(self):
        raise RuntimeError("The paper protocol has no recurrent-weight projection")


class MeasuredFeedback(LearnedFeedback):
    @torch.no_grad()
    def predict(self, u, c):
        matrix = torch.as_tensor(self.matrix, dtype=u.dtype, device=u.device)
        return slope(u)*(-c+c[:, -matrix.shape[1]:]@matrix.T)

    @torch.no_grad()
    def observe(self, u, c, z, y):
        super().observe(*(a.detach().cpu().double() for a in (u, c, z, y)))


@dataclass
class Work:
    examples: int = 0
    free_phases: int = 0
    measured_nudge_phases: int = 0
    state_iterations: int = 0
    tangent_jvps: int = 0
    homeostasis_jvps: int = 0
    homeostasis_gaussian_vectors: int = 0
    probe_scalar_reads: int = 0
    probe_excitation_sq: float = 0.

    def add(self, other):
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key)+value)


@torch.no_grad()
def free_state(model, x, steps=150):
    drive = model.drive(x)
    u = advance(torch.zeros_like(drive), drive, torch.zeros_like(drive), *model.weights(), steps)
    return u, drive


def homeostasis(model, batch, generator, probes=5, eps=None):
    if eps is None:
        eps = torch.randn(probes*batch, model.n, generator=generator,
                          dtype=model.bias.dtype, device=model.bias.device)
    with torch.enable_grad():
        je = model.recurrent(eps)-eps
        jje = model.recurrent(je)-je
        loss = (je.square().sum(-1)-(eps*jje).sum(-1)).mean()/model.n
        names, parameters = zip(*model.named_parameters())
        grads = torch.autograd.grad(loss, parameters, allow_unused=True)
    return {k: torch.zeros_like(p) if g is None else g.detach()
            for k, p, g in zip(names, parameters, grads)}, float(loss.detach())


@torch.no_grad()
def measured_responses(model, drive, u, z, beta, steps):
    """Paired fixed-current experiments, z shaped (probes, batch, states)."""
    m, batch, n = z.shape
    directions = z.reshape(m*batch, n)
    start, drive = u.repeat(m, 1), drive.repeat(m, 1)
    plus = advance(start, drive, beta*directions, *model.weights(), steps)
    minus = advance(start, drive, -beta*directions, *model.weights(), steps)
    return ((activation(plus)-activation(minus))/(2*beta)).reshape(m, batch, n)


@torch.no_grad()
def feedback(model, x, labels, method, generator, *, learner=None, beta=.01,
             free_steps=150, response_steps=20, update_learner=True):
    u, drive = free_state(model, x, free_steps)
    losses, logits, c, _ = model.cost(u, labels)
    batch = len(x)
    target = torch.nn.functional.one_hot(labels, model.outputs).to(u.dtype)
    work = Work(examples=batch, free_phases=batch, state_iterations=batch*free_steps)
    if method in ("vf_ad", "vf_ad_homeo"):
        ell = forward_tangent(u, drive, target, *model.weights(), model.readout,
                              model.readout_bias, response_steps)
        work.tangent_jvps = batch*response_steps
        work.state_iterations += batch*response_steps
    elif method == "vf_central":
        args = (u, drive, target, *model.weights(), model.readout, model.readout_bias)
        plus = advance_error(*args, beta, response_steps)
        minus = advance_error(*args, -beta, response_steps)
        ell = (activation(plus)-activation(minus))/(2*beta)
        work.measured_nudge_phases = 2*batch
        work.state_iterations += 2*batch*response_steps
    elif method == "zo_learned4":
        if learner is None:
            raise ValueError("Expected measured feedback predictor for zo_learned4; got None")
        m = 4
        baseline = learner.predict(u, c)
        z = (2*torch.randint(2, (m, batch, model.n), generator=generator,
                            device=u.device)-1).to(u.dtype)/math.sqrt(model.n)
        response = measured_responses(model, drive, u, z, beta, response_steps)
        measured = (response*c).sum(-1)
        residual = measured-(z*baseline).sum(-1)
        ell = baseline+model.n*(z*residual.unsqueeze(-1)).mean(0)
        # Fresh corrections precede predictor updates: no current-probe leakage.
        if update_learner:
            learner.observe(u, c, z, measured)
        work.measured_nudge_phases = 2*m*batch
        work.state_iterations += 2*m*batch*response_steps
        work.probe_scalar_reads = 2*m*batch
        work.probe_excitation_sq = 2*m*batch*beta**2
    else:
        raise ValueError(f"Expected method in {METHODS}; got {method}")
    residual = model.force(u, drive).norm(dim=-1)
    stats = dict(loss=float(losses.mean()), accuracy=float((logits.argmax(-1)==labels).float().mean()),
                 free_residual_mean=float(residual.mean()), free_residual_max=float(residual.max()))
    return ell, u, work, stats


def training_gradient(model, x, labels, method, generator, *, homeo_generator=None, **kwargs):
    ell, u, work, stats = feedback(model, x, labels, method, generator, **kwargs)
    grads = local_gradient(model, x, u, labels, ell)
    if method == "vf_ad_homeo":
        reg, value = homeostasis(model, len(x), homeo_generator)
        grads = {k: value_g+reg[k] for k, value_g in grads.items()}
        stats["homeostasis_loss"] = value
        work.homeostasis_jvps = 10*len(x)
        work.homeostasis_gaussian_vectors = 5*len(x)
    return grads, work, stats


@torch.no_grad()
def diagnostics(model, x, labels, method, learner, free_steps=150, response_steps=20, beta=.01):
    """Read-only, double precision oracle on a fixed training cohort."""
    import copy
    ref = copy.deepcopy(model).cpu().double()
    x, labels = x.cpu().double(), labels.cpu()
    prediction = copy.deepcopy(learner)
    ell, u, _, _ = feedback(ref, x, labels, method, torch.Generator().manual_seed(71000),
        learner=prediction, beta=beta, free_steps=free_steps, response_steps=response_steps,
        update_learner=False)
    c = ref.cost(u, labels)[2]
    w = ref.dense_w()
    d = slope(u)
    ju = w*d.unsqueeze(-2)-torch.eye(ref.n, dtype=u.dtype)
    truth = torch.linalg.solve(ju.transpose(-1, -2), (d*c).unsqueeze(-1)).squeeze(-1)
    cosine = lambda a,b: float(torch.nn.functional.cosine_similarity(a,b,dim=-1,eps=1e-30).mean())
    error = lambda a,b: float((a-b).norm()/b.norm().clamp_min(1e-30))
    j = w-torch.eye(ref.n, dtype=u.dtype)
    symmetric, asymmetric = (j+j.T)/2, (j-j.T)/2
    u_long, _ = free_state(ref, x, 2*free_steps)
    q = forward_tangent(u, ref.drive(x), torch.nn.functional.one_hot(labels,ref.outputs).double(),
                        *ref.weights(), ref.readout, ref.readout_bias, response_steps)
    q_long = forward_tangent(u, ref.drive(x), torch.nn.functional.one_hot(labels,ref.outputs).double(),
                             *ref.weights(), ref.readout, ref.readout_bias, 2*response_steps)
    z = (2*torch.randint(2,(4,len(u),ref.n),generator=torch.Generator().manual_seed(72000))-1).double()/math.sqrt(ref.n)
    probe_values = (measured_responses(ref,ref.drive(x),u,z,beta,response_steps)*c).sum(-1)
    exact_values = (z*truth).sum(-1)
    layer_cos = {name:cosine(a,b) for name,a,b in zip(("hidden1","hidden2","output"),
                    ell.split(ref.sizes,dim=-1), truth.split(ref.sizes,dim=-1))}
    task_g = local_gradient(ref,x,u,labels,ell)
    true_g = local_gradient(ref,x,u,labels,truth)
    gradients = {k:dict(cosine=cosine(task_g[k].flatten()[None],true_g[k].flatten()[None]),
                        relative_error=error(task_g[k],true_g[k])) for k in task_g}
    feedback_pairs = torch.cat((ref.backward1.flatten(), ref.backward2.flatten()))
    forward_pairs = torch.cat((ref.forward1.T.flatten(), ref.forward2.T.flatten()))
    return dict(jacobian_symmetry=float(symmetric.norm()/(symmetric.norm()+asymmetric.norm())),
        jacobian_asymmetry_norm=float(asymmetric.norm()),
        weight_angle_degrees=math.degrees(math.acos(max(-1.,min(1.,cosine(forward_pairs[None],feedback_pairs[None]))))),
        feedback_cosine=cosine(ell,truth), feedback_relative_error=error(ell,truth),
        layer_feedback_cosine=layer_cos, parameter_task_gradients=gradients,
        free_150_vs_300_max=float((u-u_long).abs().max()),
        response_20_vs_40_relative_error=error(q,q_long),
        measured_probe_projection_relative_error=error(probe_values,exact_values),
        free_residual_max=float(ref.force(u,ref.drive(x)).norm(dim=-1).max()),
        oracle_equation_residual=float((ju.transpose(-1,-2)@truth.unsqueeze(-1)-(d*c).unsqueeze(-1)).abs().max()))
