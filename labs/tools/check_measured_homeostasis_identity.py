"""Algebra audit of a clamped-force alternative to the mean homeostatic update.

This is an eight-state, noiseless identity check, not a calibration/training
experiment or a claim about hardware access. The measurement callback exposes
force values at clamped activities. It is stronger access than DC current nudges.
"""

import argparse
import json
import math
from pathlib import Path

import torch

from labs.fmnist_homeostasis import PaperNet, homeostasis


@torch.no_grad()
def measured_gradient(force, activity, directions, amplitude):
    """Unit-norm directions with mean outer product I/n give the mean gradient.

    Only force values, clamped activities, and the known probe design enter
    this calculation. No force derivative or transpose application is used.
    """
    plus = force(activity + amplitude * directions)
    minus = force(activity - amplitude * directions)
    response = (plus - minus) / (2 * amplitude)
    return 2 * (response.T @ directions - directions.T @ response) / len(directions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f'Expected fresh output path; got {args.output}')
    torch.set_num_threads(1)
    model = PaperNet(inputs=5, hidden=3, outputs=2, seed=20260914, dtype=torch.float64)
    n, h = model.n, model.hidden
    activity = torch.linspace(.2, .8, n, dtype=torch.float64)
    drive = torch.linspace(-.3, .4, n, dtype=torch.float64)

    @torch.no_grad()
    def force(r):
        # G(r)=W r+drive-phi^{-1}(r); its zeros are the original force zeros.
        # On hardware this would require activity clamping and current readout.
        if not bool(((r > 0) & (r < 1)).all()):
            raise ValueError('Expected clamped sigmoid activities inside (0, 1)')
        return model.recurrent(r) + drive - (torch.logit(r) + 2) / 4

    hadamard = torch.ones((1, 1), dtype=torch.float64)
    while len(hadamard) < n:
        hadamard = torch.cat((torch.cat((hadamard, hadamard), 1),
                             torch.cat((hadamard, -hadamard), 1)), 0)
    directions = hadamard / math.sqrt(n)
    assert torch.allclose(directions.T @ directions, torch.eye(n, dtype=torch.float64))

    # Oracle references are confined to the audit, outside measured_gradient.
    w = model.dense_w().detach()
    analytic = 2 * (w - w.T) / n
    ad, _ = homeostasis(model, 1, None, eps=hadamard)
    blocks = dict(forward1=(slice(h, 2*h), slice(0, h)),
                  backward1=(slice(0, h), slice(h, 2*h)),
                  forward2=(slice(2*h, n), slice(h, 2*h)),
                  backward2=(slice(h, 2*h), slice(2*h, n)))
    cases = {}
    for amplitude in (.01, .001, .0001):
        measured = measured_gradient(force, activity, directions, amplitude)
        errors = {name: float((measured[indices] - ad[name]).abs().max())
                  for name, indices in blocks.items()}
        error = float((measured - analytic).abs().max())
        assert error < 1e-10 and max(errors.values()) < 1e-10, (amplitude, error, errors)
        cases[str(amplitude)] = dict(max_abs_error_to_analytic_mean=error,
                                    max_abs_error_to_author_rule_by_block=errors)

    result = dict(status='complete', states=n, hidden=h, outputs=model.outputs,
        model_seed=20260914, probe_directions=n, clamped_configurations=2*n,
        vector_force_reads=2*n, scalar_current_reads=2*n*n,
        probe_norm=1, oracle_reference='Expected paper-code loss gradient; full orthogonal probe design',
        cases=cases, limitations=[
            'No calibration or task training, measurement noise, or convergence-rate test.',
            'The complete orthogonal design checks the mean update, not samplewise equality to the stochastic AD rule.',
            'State clamping and force/current readout are assumed; equilibrium voltage response Rz cannot replace Jz.',
            'Activities are selected inside the sigmoid range, not taken from a Fashion-MNIST equilibrium.'
        ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
