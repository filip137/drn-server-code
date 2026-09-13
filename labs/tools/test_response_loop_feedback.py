"""Small paired-DC calibration audit on independently wired physical skew loops.

No task training is launched. The measurement interface reads one scalar
projection per perturbed equilibrium. Known skew enters post-update diagnostics
only. The equilibrium solver implements the physical model and uses no adjoint.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from labs.recurrent_eqprop import settle
from labs.response_loop_feedback import calibrate_response_loops
from labs.tools.run_random_nudge_hopfield import write_csv, write_json
from labs.tools.test_circulation_feedback import audit_controller
from labs.tools.test_structured_circulation import make_loop_case
from labs.tools.train_recurrent_eqprop_digits import dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sizes', nargs='+', type=int, default=[64, 256])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--loops', type=int, default=4)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--learning-rate', type=float, default=0.2)
    parser.add_argument('--amplitude', type=float, default=0.01)
    parser.add_argument('--read-noise', type=float, default=1e-5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Expected fresh output path; got {args.output}')
    if args.amplitude <= 0 or args.read_noise < 0:
        parser.error('Expected positive amplitude and nonnegative read noise')
    args.output.mkdir(parents=True)
    config = dict(vars(args), output=str(args.output))
    write_json(args.output/'run.json', dict(config=config,
        command=[sys.executable, '-m', 'labs.tools.test_response_loop_feedback', *sys.argv[1:]],
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_status=subprocess.check_output(['git', 'status', '--short'], text=True),
        expected_cases=len(args.sizes)*len(args.seeds), evidence_tier='exploratory calibration only'))
    started, results = time.time(), []
    data = dataset()
    write_json(args.output/'status.json', dict(status='running', completed=0))
    for size in args.sizes:
        for seed in args.seeds:
            model, left, right, reference = make_loop_case(seed, size, args.loops)
            network = model.network()
            drive = model.drive(data['train_x'][:1])[0]
            free = settle(network, drive)
            center = free.state
            directions = np.concatenate((left.T, right.T))
            readouts = np.concatenate((right.T, left.T))
            rng = np.random.default_rng(72000+seed)
            history = []

            def matrix(coefficients):
                forward = (left*coefficients)@right.T
                return (forward-forward.T)/np.sqrt(2)

            def measure(coefficients):
                controller = matrix(coefficients)
                kw = dict(initial=center, skew_correction=-controller/2,
                          center=center, tolerance=1e-9)
                # F+C(s-s0)-b=0; the existing drive uses the opposite sign.
                positive = settle(network, drive-args.amplitude*directions, **kw)
                negative = settle(network, drive+args.amplitude*directions, **kw)
                # Full arrays are simulator state; only these scalar readouts
                # are exposed to the measurement/calibration interface.
                yp = np.sum((positive.state-center)*readouts, axis=-1)
                yn = np.sum((negative.state-center)*readouts, axis=-1)
                yp += rng.normal(scale=args.read_noise, size=yp.shape)
                yn += rng.normal(scale=args.read_noise, size=yn.shape)
                response = (yp-yn)/(2*args.amplitude)
                defect = np.sqrt(2)*(response[:args.loops]-response[args.loops:])
                return defect, dict(equilibrations=positive.equilibrations+negative.equilibrations,
                    scalar_reads=4*args.loops,
                    total_excitation_sq=4*args.loops*args.amplitude**2,
                    relaxation_iterations=positive.iterations+negative.iterations)

            def audit(row, coefficients):
                row['controller_relative_error'] = float(np.linalg.norm(
                    coefficients+reference['normalized_gains'])/np.linalg.norm(reference['normalized_gains']))
                history.append(row)

            coefficients, counts, _ = calibrate_response_loops(measure, args.loops,
                steps=args.steps, learning_rate=args.learning_rate, callback=audit)
            controller = matrix(coefficients)
            case = args.output/f'n{size}_seed{seed}'
            case.mkdir()
            write_csv(case/'calibration.csv', history)
            np.savez_compressed(case/'controller.npz', controller=controller,
                                coefficients=coefficients, left=left, right=right)
            result = dict(size=size, seed=seed, loops=args.loops,
                controller_relative_error=history[-1]['controller_relative_error'],
                calibration_counts=counts, free_anchor_equilibrations=free.equilibrations,
                anchor_projection_reads=2*args.loops,
                known_wiring_coefficients=2*size*args.loops,
                learned_coefficients=args.loops, process_noise_channels=0,
                gradient_audit=audit_controller(model, data, controller),
                task_training_performed=False)
            assert counts['equilibrations'] == 4*args.loops*args.steps
            assert counts['scalar_reads'] == 4*args.loops*args.steps
            write_json(case/'summary.json', result)
            results.append(result)
            write_json(args.output/'summary.json', results)
            write_json(args.output/'status.json', dict(status='running', completed=len(results),
                elapsed_seconds=time.time()-started))
            print(f'n={size} seed={seed}: controller error {result["controller_relative_error"]:.4g}', flush=True)
    write_json(args.output/'status.json', dict(status='complete', completed=len(results),
        elapsed_seconds=time.time()-started))


if __name__ == '__main__':
    main()
