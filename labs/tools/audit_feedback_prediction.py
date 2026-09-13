#!/usr/bin/env python3
"""Read-only comparison of saved feedback predictions and fresh MC corrections.

This is an oracle linear-response diagnostic, NOT new physical training. The
saved predictors were trained with four probes. It does not establish that
training with fewer probes or no correction succeeds from initialization.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

from labs.adjoint_baselines import MeasuredBaseline, local_slope_baseline
from labs.adjoint_estimators import estimate_adjoint, make_probes
from labs.recurrent_eqprop import Classifier, cost_gradient, flatten_gradient, parameter_gradient, settle
from labs.tools.run_random_nudge_hopfield import cosine, write_json


def vector_measurement_identity():
    """Verify the proposed vector-readout fit on a stable linear example."""
    rng = np.random.default_rng(83001)
    n, outputs = 16, 3
    diagonal = rng.uniform(1, 2, size=n)
    raw = rng.normal(size=(n, n))
    symmetric = (raw+raw.T)/2
    symmetric *= .4/np.linalg.norm(symmetric, 2)
    skew = (raw-raw.T)/2
    skew *= .8/np.linalg.norm(skew, 2)
    jacobian = symmetric+skew-np.diag(diagonal)
    selector = np.eye(n)[-outputs:]
    probes = make_probes(rng, n, n, "hadamard")
    # Known linear oracle responses used solely to verify the algebra.
    responses = np.linalg.solve(jacobian, probes.T).T
    output_responses = responses[:, -outputs:]
    a = probes/diagonal
    target = output_responses+a[:, -outputs:]
    fitted, *_ = np.linalg.lstsq(a, target, rcond=None)
    exact_map = np.linalg.solve(jacobian.T, selector.T)
    exact_h = diagonal[:, None]*exact_map+selector.T
    recovered_map = (fitted-selector.T)/diagonal[:, None]
    error = float(np.linalg.norm(recovered_map-exact_map))
    assert error < 1e-12
    current = rng.normal(size=fitted.shape)
    k, rate = 3, .25
    residual = target[k]-a[k] @ current
    updated = current+rate*np.outer(a[k], residual)/(a[k] @ a[k])
    np.testing.assert_allclose(target[k]-a[k] @ updated, (1-rate)*residual, atol=1e-12)
    return dict(kind="derived algebra check, stable 16-state linear system", states=n,
                outputs=outputs, probe_directions=n, map_error=error,
                h_equation_error=float(np.linalg.norm(a @ exact_h-target)),
                vector_projection_residual_factor=1-rate)


def audit(checkpoint, trials, seed):
    with np.load(checkpoint) as v:
        model = Classifier(v["symmetric"],v["skew"],v["inputs"],v["bias"],
                           outputs=int(v["outputs"]),cubic=float(v["cubic"]),
                           logit_scale=float(v["logit_scale"]),symmetric_cap=float(v["symmetric_cap"]))
        x, labels = v["val_x"][:96], v["val_y"][:96]
        predictor = MeasuredBaseline(v["predictor_matrix"].copy())
        observations = int(v["predictor_observations"])
    net = model.network()
    free = settle(net, model.drive(x)).state
    c = cost_gradient(free,labels,model.hidden,model.logit_scale)
    jacobian = net.jacobian(free)
    exact = np.linalg.solve(np.swapaxes(jacobian,-1,-2),c[...,None])[...,0]
    reference = flatten_gradient(parameter_gradient(model,x,free,exact))
    reference_norm = np.linalg.norm(reference)
    baseline = predictor.predict(free,c,model.cubic)

    def quality(feedback):
        gradient = flatten_gradient(parameter_gradient(model,x,free,feedback))
        return dict(gradient_cosine=cosine(gradient,reference),
                    gradient_relative_error=float(np.linalg.norm(gradient-reference)/reference_norm),
                    adjoint_relative_error=float(np.linalg.norm(feedback-exact)/np.linalg.norm(exact)))

    result = dict(checkpoint=str(checkpoint.resolve()), audit_seed=seed, states=model.size,
                  validation_examples=len(x), previous_predictor_observations=observations,
                  local_baseline=quality(local_slope_baseline(free,c,model.cubic)),
                  learned_baseline=quality(baseline), mc=[])
    rng = np.random.default_rng(seed)
    stats = {m: [] for m in (1,2,4)}
    for _ in range(trials):
        probes = (2*rng.integers(0,2,size=(len(x),4,model.size))-1)/np.sqrt(model.size)
        readings = np.einsum("bmn,bn->bm",probes,exact)
        # Same independent paired-voltage read-noise model as training. No
        # finite-amplitude or settling bias is represented in this audit.
        readings += rng.normal(size=readings.shape)*np.linalg.norm(c,axis=-1)[:,None]*1e-5/(np.sqrt(2)*.01)
        for m in stats:
            stats[m].append(quality(estimate_adjoint(baseline,probes[:,:m],readings[:,:m],"mc")))
    for m, values in stats.items():
        entry = dict(probes=m,trials=trials)
        for metric in values[0]:
            entry[metric+"_mean"] = float(np.mean([v[metric] for v in values]))
            entry[metric+"_std"] = float(np.std([v[metric] for v in values]))
        result["mc"].append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints",type=Path,nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--trials",type=int,default=64)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Expected a fresh output path; got existing path {args.output}")
    if args.trials < 1:
        parser.error(f"Expected a positive trial count; got {args.trials}")
    rows=[]
    for i,path in enumerate(args.checkpoints):
        rows.append(audit(path,args.trials,84000+i))
        print(f"Audited {path.name}: learned-only gradient cosine={rows[-1]['learned_baseline']['gradient_cosine']:.4f}",flush=True)
    algebra = vector_measurement_identity()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,dict(kind="read-only frozen-checkpoint oracle linear-response diagnostic",
                               command=[sys.executable,*sys.argv], trials=args.trials,
                               nudge_amplitude_for_noise=.01,voltage_read_noise=1e-5,
                               predictor_trained_with_probes=4,
                               finite_nudge_bias_included=False,parameter_updates_performed=False,
                               checkpoints=rows,vector_measurement_identity=algebra))
    print(f"Saved: {args.output}",flush=True)


if __name__ == "__main__":
    main()
