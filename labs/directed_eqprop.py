"""Untied layered vector-field EqProp and the published-code homeostasis rule.

Raw membrane dynamics: F(u) = W phi(u) + Bx + b - u.
Equivalent activity equilibrium force: G(r) = Wr + Bx + b - phi^-1(r).
Nudges use F - external_current = 0. These two choices of cost gradient and
measured response are deliberately separate; see docs/directed_eqprop_mnist.md.
All physical estimators use settled states, without Jacobians or transpose solves.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch


METHODS = ("vf_membrane", "ep_activity", "ep_homeo", "ep_probe4",
           "ep_probe4_homeo", "learned_probe4", "learned_probe4_homeo", "adjoint")


def activation(u):
    return torch.sigmoid(4*u-2)


def slope(u):
    r = activation(u)
    return 4*r*(1-r)


class LearnedFeedback:
    """Measured online feedback predictor D(-c + H c_out), with no oracle data.

    Scalar normalized-LMS updates use the same fresh response measurements,
    after the current gradient's unbiased residual correction has been formed.
    This adapts the earlier learned baseline to sigmoid activity coordinates.
    """

    def __init__(self, n, outputs, rate=.25):
        self.matrix=np.zeros((n,outputs))
        self.rate=rate
        self.observations=0

    @torch.no_grad()
    def predict(self, u, c):
        return slope(u)*(-c+c[:,-self.matrix.shape[1]:]@torch.from_numpy(self.matrix).T)

    @torch.no_grad()
    def observe(self, u, c, z, y):
        d=slope(u).numpy()
        c=c.numpy()
        z,y=z.numpy(),y.numpy()
        outputs=self.matrix.shape[1]
        for example in range(len(u)):
            e=c[example,-outputs:]
            for probe in range(len(z)):
                a=z[probe,example]*d[example]
                target=y[probe,example]+a@c[example]
                residual=target-a@self.matrix@e
                denominator=(a@a)*(e@e)
                if denominator>1e-30:
                    self.matrix+=self.rate*residual/denominator*np.outer(a,e)
                self.observations+=1


class DirectedNet(torch.nn.Module):
    """Two hidden layers, a dynamical output layer and a trained linear readout.

    Each directed matrix is a distinct parameter, even at alpha=0. A common
    recurrent norm cap guarantees contraction; it never ties matrix entries.
    """

    def __init__(self, inputs=784, hidden=64, outputs=10, seed=0, alpha=90., cap=.95):
        super().__init__()
        self.inputs, self.hidden, self.outputs, self.cap = inputs, hidden, outputs, cap
        self.sizes = (hidden, hidden, outputs)
        self.n = sum(self.sizes)
        gen = torch.Generator().manual_seed(seed)
        rand = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float64)
        f1 = rand(hidden, hidden)/math.sqrt(hidden)
        f2 = rand(outputs, hidden)/math.sqrt(hidden)
        # Independent reverse draws; same entry variance as each corresponding
        # forward matrix, so the mixture does not change the expected pair norm.
        g1 = rand(hidden, hidden)/math.sqrt(hidden)
        g2 = rand(hidden, outputs)/math.sqrt(hidden)
        # An alpha-independent upper bound keeps the forward weights identical
        # across the asymmetry sweep, without angle-dependent normalization.
        norm2 = lambda a: float(torch.linalg.matrix_norm(a, ord=2))**2
        bound = math.sqrt(norm2(f1)+norm2(f2)+max(norm2(g1), norm2(g2)))
        scale = cap/bound
        f1, f2, g1, g2 = (a*scale for a in (f1, f2, g1, g2))
        b1 = math.cos(math.radians(alpha))*f1.T + math.sin(math.radians(alpha))*g1
        b2 = math.cos(math.radians(alpha))*f2.T + math.sin(math.radians(alpha))*g2
        self.input = torch.nn.Parameter(.35*rand(hidden, inputs)/math.sqrt(inputs))
        self.forward1 = torch.nn.Parameter(f1)
        self.forward2 = torch.nn.Parameter(f2)
        self.backward1 = torch.nn.Parameter(b1)
        self.backward2 = torch.nn.Parameter(b2)
        self.bias = torch.nn.Parameter(torch.zeros(self.n, dtype=torch.float64))
        self.readout = torch.nn.Parameter(rand(outputs, outputs)/math.sqrt(outputs))
        self.readout_bias = torch.nn.Parameter(torch.zeros(outputs, dtype=torch.float64))
        self.project()
        with torch.no_grad():
            # Centre the zero-input fixed point at phi(.5)=.5.
            self.bias.copy_(.5-self.recurrent(torch.full((1, self.n), .5, dtype=torch.float64))[0])

    def recurrent(self, r):
        a, b, c = r.split(self.sizes, dim=-1)
        return torch.cat((b@self.backward1.T,
                          a@self.forward1.T+c@self.backward2.T,
                          b@self.forward2.T), dim=-1)

    def recurrent_transpose(self, v):
        a, b, c = v.split(self.sizes, dim=-1)
        return torch.cat((b@self.forward1,
                          a@self.backward1+c@self.forward2,
                          b@self.backward2), dim=-1)

    def drive(self, x):
        d = self.bias.expand(len(x), -1).clone()
        d[:, :self.hidden] += x@self.input.T
        return d

    def force(self, u, drive):
        return self.recurrent(activation(u))+drive-u

    def cost(self, u, labels):
        r = activation(u)
        logits = r[:, -self.outputs:]@self.readout.T+self.readout_bias
        errors = logits.softmax(-1)
        errors = errors-torch.nn.functional.one_hot(labels, self.outputs)
        c = torch.zeros_like(u)
        c[:, -self.outputs:] = errors@self.readout
        losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        return losses, logits, c, errors

    def dense_w(self):
        # Oracle diagnostics only. Physical methods never call this function.
        return self.recurrent(torch.eye(self.n, dtype=torch.float64)).T

    def jacobian(self, u, coordinate="membrane"):
        w = self.dense_w()
        if coordinate == "code":
            return w-torch.eye(self.n, dtype=u.dtype)
        d = slope(u)
        if coordinate == "membrane":
            return w*d.unsqueeze(-2)-torch.eye(self.n, dtype=u.dtype)
        if coordinate == "activity":
            return w-torch.diag_embed(d.reciprocal())
        raise ValueError(f"Expected membrane, activity, or code coordinates; got {coordinate}")

    @torch.no_grad()
    def project(self):
        # Bipartite layer structure: these two rectangular blocks have the
        # same nonzero singular values as the complete recurrent operator.
        top = torch.cat((self.backward1, self.forward2), dim=0)
        bottom = torch.cat((self.forward1, self.backward2), dim=1)
        norm = max(float(torch.linalg.matrix_norm(top, ord=2)),
                   float(torch.linalg.matrix_norm(bottom, ord=2)))
        factor = min(1., self.cap/max(norm, 1e-30))
        for p in (self.forward1, self.forward2, self.backward1, self.backward2):
            p.mul_(factor)
        return factor


@dataclass
class Counts:
    equilibrations: int = 0
    state_reads: int = 0
    scalar_reads: int = 0
    relaxation_state_iterations: int = 0
    oracle_adjoint_iterations: int = 0
    homeostasis_gaussian_vectors: int = 0
    homeostasis_jvps: int = 0
    excitation_sq: float = 0.
    max_residual: float = 0.

    def add(self, other):
        for name in self.__dataclass_fields__:
            old, new = getattr(self, name), getattr(other, name)
            setattr(self, name, max(old, new) if name == "max_residual" else old+new)


@torch.no_grad()
def settle(model, drive, *, initial=None, current=None, beta=0., labels=None,
           coordinate="activity", tolerance=1e-10, max_steps=600, counts=None):
    """Synchronous fixed point relaxation, including state-dependent error nudges.

    Reject incomplete relaxation; every returned state satisfies the force
    residual tolerance. Probe currents are fixed; error nudges track the cost.
    """
    u = drive.clone() if initial is None else initial.clone()
    for iteration in range(1, max_steps+1):
        new = model.recurrent(activation(u))+drive
        if current is not None:
            new = new-current
        if beta:
            c = model.cost(u, labels)[2]
            if coordinate == "membrane":
                c = c*slope(u)
            new = new-beta*c
        residual = float((new-u).abs().max())
        if not math.isfinite(residual):
            raise RuntimeError("Non-finite equilibrium residual")
        if residual <= tolerance:
            if counts is not None:
                counts.equilibrations += len(u)
                counts.relaxation_state_iterations += len(u)*iteration
                counts.max_residual = max(counts.max_residual, residual)
            return u
        u = new
    raise RuntimeError(f"Equilibrium failed after {max_steps} steps: residual={residual:g}")


@torch.no_grad()
def adjoint(model, u, c_activity, tolerance=1e-12, counts=None):
    """Digital reference: solve (W D-I)^T lambda = D c_r by iteration."""
    d = slope(u)
    c = d*c_activity
    ell = -c
    for iteration in range(1, 1000):
        new = d*model.recurrent_transpose(ell)-c
        if float((new-ell).abs().max()) <= tolerance:
            if counts is not None:
                counts.oracle_adjoint_iterations += iteration*len(u)
            return ell
        ell = new
    raise RuntimeError("Adjoint reference did not converge")


@torch.no_grad()
def local_gradient(model, x, u, labels, ell):
    """-partial_theta F^T ell, plus the explicit readout-cost gradient.

    Independent synapses use presynaptic free activity times the postsynaptic
    feedback signal. No symmetric energy contrast and no tied-weight update.
    """
    r1, r2, ro = activation(u).split(model.sizes, dim=-1)
    l1, l2, lo = ell.split(model.sizes, dim=-1)
    errors = model.cost(u, labels)[3]
    batch = len(x)
    return dict(input=-l1.T@x/batch, forward1=-l2.T@r1/batch,
                forward2=-lo.T@r2/batch, backward1=-l1.T@r2/batch,
                backward2=-l2.T@ro/batch, bias=-ell.mean(0),
                readout=errors.T@ro/batch, readout_bias=errors.mean(0))


def homeostasis(model, batch_size, generator, probes=5, eps=None):
    """Authors' activation-bypassed JVP penalty, differentiated in parameters.

    J_code = W-I, so the two exact forward JVPs are available as matvecs.
    Gaussian probes are independent per sample. This is digital AD work,
    not a physical nudge experiment. E[L] = 2 ||A_W||_F^2 / n.
    """
    if eps is None:
        eps = torch.randn(probes*batch_size, model.n, generator=generator, dtype=torch.float64)
    with torch.enable_grad():
        j_eps = model.recurrent(eps)-eps
        jj_eps = model.recurrent(j_eps)-j_eps
        loss = (j_eps.square().sum(-1)-(eps*jj_eps).sum(-1)).mean()/model.n
        names, params = zip(*model.named_parameters())
        derivatives = torch.autograd.grad(loss, params, allow_unused=True)
    grads = {name: torch.zeros_like(p) if g is None else g.detach()
             for name, p, g in zip(names, params, derivatives)}
    return grads, float(loss.detach())


@torch.no_grad()
def paired_response(model, drive, free, directions, *, beta, tolerance, counts,
                    coordinate="activity", noise=0., generator=None):
    """One paired equilibrium response per row; direction norms are explicit."""
    plus = settle(model, drive, initial=free, current=beta*directions,
                  tolerance=tolerance, counts=counts)
    minus = settle(model, drive, initial=free, current=-beta*directions,
                   tolerance=tolerance, counts=counts)
    if coordinate == "activity":
        plus, minus = activation(plus), activation(minus)
    response = (plus-minus)/(2*beta)
    if noise:
        response += noise/(math.sqrt(2)*beta)*torch.randn(response.shape, generator=generator, dtype=response.dtype)
    counts.state_reads += 2*len(free)
    counts.scalar_reads += 2*len(free)*model.n
    counts.excitation_sq += 2*beta**2*float(directions.square().sum())
    return response


@torch.no_grad()
def feedback(model, x, labels, method, generator, *, beta=.01, tolerance=1e-10,
             noise=0., probes=4, learner=None, update_learner=True):
    if method not in METHODS:
        raise ValueError(f"Expected one of {METHODS}; got {method}")
    counts = Counts()
    drive = model.drive(x)
    free = settle(model, drive, tolerance=tolerance, counts=counts)
    counts.state_reads += len(x)
    counts.scalar_reads += len(x)*model.n
    losses, logits, c, _ = model.cost(free, labels)
    if method == "adjoint":
        ell = adjoint(model, free, c, counts=counts)
    elif method.startswith("learned"):
        if learner is None:
            raise ValueError(f"Expected a LearnedFeedback instance for {method}; got None")
        ell=learner.predict(free,c)
    else:
        coordinate = "membrane" if method == "vf_membrane" else "activity"
        plus = settle(model, drive, initial=free, beta=beta, labels=labels,
                      coordinate=coordinate, tolerance=tolerance, counts=counts)
        minus = settle(model, drive, initial=free, beta=-beta, labels=labels,
                       coordinate=coordinate, tolerance=tolerance, counts=counts)
        # Actual currents depend on the nudged equilibrium, record both norms.
        for sign_u in (plus, minus):
            sign_c = model.cost(sign_u, labels)[2]
            if coordinate == "membrane":
                sign_c = sign_c*slope(sign_u)
            counts.excitation_sq += beta**2*float(sign_c.square().sum())
        if coordinate == "activity":
            plus, minus = activation(plus), activation(minus)
        ell = (plus-minus)/(2*beta)
        if noise:
            ell += noise/(math.sqrt(2)*beta)*torch.randn(ell.shape, generator=generator, dtype=ell.dtype)
        counts.state_reads += 2*len(x)
        counts.scalar_reads += 2*len(x)*model.n
    if "probe" in method:
        # Unit-norm random signs; batch order [probe, example, state].
        z = (2*torch.randint(2, (probes, len(x), model.n), generator=generator)-1).to(torch.float64)/math.sqrt(model.n)
        rz = paired_response(model, drive.repeat(probes, 1), free.repeat(probes, 1),
                             z.reshape(-1, model.n), beta=beta, tolerance=tolerance,
                             counts=counts, noise=noise, generator=generator)
        rz = rz.reshape_as(z)
        y = (rz*c.unsqueeze(0)).sum(-1)
        residual = y-(z*ell.unsqueeze(0)).sum(-1)
        ell = ell+model.n*(z*residual.unsqueeze(-1)).mean(0)
        if learner is not None and update_learner:
            learner.observe(free,c,z,y)
    return ell, free, counts, dict(loss=float(losses.mean()),
                                  accuracy=float((logits.argmax(-1)==labels).double().mean()))


def training_gradient(model, x, labels, method, generator, *, homeo_generator=None,
                      homeo_coefficient=1., **kwargs):
    ell, free, counts, metrics = feedback(model, x, labels, method, generator, **kwargs)
    gradient = local_gradient(model, x, free, labels, ell)
    if "homeo" in method:
        reg, loss = homeostasis(model, len(x), homeo_generator or generator)
        gradient = {k: g+homeo_coefficient*reg[k] for k, g in gradient.items()}
        counts.homeostasis_gaussian_vectors += 5*len(x)
        counts.homeostasis_jvps += 10*len(x)
        metrics["homeostasis_sample_loss"] = loss
    return gradient, counts, metrics


def cosine(a, b):
    a, b = a.flatten(), b.flatten()
    return float(a.dot(b)/(a.norm()*b.norm()).clamp_min(1e-30))


@torch.no_grad()
def audit(model, x, labels, method, seed=707, **kwargs):
    """Read-only matched cohort; reference work excluded from training counts."""
    ell, u, counts, _ = feedback(model, x, labels, method,
                                torch.Generator().manual_seed(seed), update_learner=False, **kwargs)
    c = model.cost(u, labels)[2]
    truth = adjoint(model, u, c)
    reference = local_gradient(model, x, u, labels, truth)
    estimate = local_gradient(model, x, u, labels, ell)
    relative = lambda a, b: float((a-b).norm()/b.norm().clamp_min(1e-30))
    result = dict(feedback_cosine=cosine(ell, truth), feedback_relative_error=relative(ell, truth),
                  audit_counts=counts.__dict__)
    for name, e, t in zip(("hidden1", "hidden2", "output"), ell.split(model.sizes, -1), truth.split(model.sizes, -1)):
        per_sample = (e*t).sum(-1)/(e.norm(dim=-1)*t.norm(dim=-1)).clamp_min(1e-30)
        result[name+"_feedback_cosine"] = float(per_sample.mean())
        result[name+"_feedback_relative_error"] = relative(e, t)
    for name in reference:
        result[name+"_gradient_cosine"] = cosine(estimate[name], reference[name])
        result[name+"_gradient_relative_error"] = relative(estimate[name], reference[name])
    for coord in ("code", "membrane"):
        j = model.jacobian(u, coord)
        s, a = (j+j.transpose(-1, -2))/2, (j-j.transpose(-1, -2))/2
        sn, an = s.norm(dim=(-2,-1)), a.norm(dim=(-2,-1))
        result[coord+"_antisymmetry_norm"] = float(an.mean())
        result[coord+"_symmetry_score"] = float((sn/(sn+an)).mean())
    for name, f, b in (("pair1", model.forward1, model.backward1),
                       ("pair2", model.forward2, model.backward2)):
        result[name+"_weight_angle_degrees"] = math.degrees(math.acos(max(-1., min(1., cosine(f, b.T)))))
    d = slope(u)
    result["activation_slope_min"] = float(d.min())
    result["activation_slope_mean"] = float(d.mean())
    result["saturated_fraction"] = float((d<.01).double().mean())
    return result
