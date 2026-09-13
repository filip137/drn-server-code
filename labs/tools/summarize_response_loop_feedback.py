"""Verify and plot fixed-budget DC loop calibration audits (no task training)."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from labs.tools.run_random_nudge_hopfield import write_json
from labs.tools.test_structured_circulation import make_loop_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f'Expected fresh output path; got {args.output}')
    args.output.mkdir(parents=True)
    rows, traces = [], []
    for source in args.runs:
        run = json.loads((source/'run.json').read_text())
        config = run['config']
        status = json.loads((source/'status.json').read_text())
        cases = json.loads((source/'summary.json').read_text())
        assert status['status'] == 'complete'
        assert len(cases) == status['completed'] == run['expected_cases']
        assert {(c['size'], c['seed']) for c in cases} == {
            (n, seed) for n in config['sizes'] for seed in config['seeds']}
        for case in cases:
            folder = source/f"n{case['size']}_seed{case['seed']}"
            saved = np.load(folder/'controller.npz')
            model, left, right, _ = make_loop_case(case['seed'], case['size'], case['loops'])
            assert np.array_equal(saved['left'], left) and np.array_equal(saved['right'], right)
            product = (left*saved['coefficients'])@right.T
            controller = (product-product.T)/np.sqrt(2)
            assert np.allclose(controller, saved['controller'])
            relative = np.linalg.norm(controller+model.skew)/np.linalg.norm(model.skew)
            assert np.isclose(relative, case['controller_relative_error'])
            expected = 4*case['loops']*config['steps']
            assert case['calibration_counts']['equilibrations'] == expected
            assert case['calibration_counts']['scalar_reads'] == expected
            with (folder/'calibration.csv').open() as stream:
                trace = list(csv.DictReader(stream))
            assert len(trace) == config['steps']
            assert np.isclose(float(trace[-1]['controller_relative_error']), relative)
            audit = next(a for a in case['gradient_audit'] if a['method']=='calibrated_double_gain')
            rows.append(dict(size=case['size'], seed=case['seed'], steps=config['steps'],
                controller_relative_error=float(relative), gradient_cosine=audit['gradient_cosine'],
                input_gradient_cosine=audit['input_gradient_cosine'], perturbed_equilibrations=expected,
                scalar_reads=expected, free_anchor_equilibrations=1, anchor_projection_reads=2*case['loops']))
            traces.append((case['size'], config['steps'], case['loops'], trace))
    fig, ax = plt.subplots(figsize=(6.4, 4.0), constrained_layout=True)
    longest = max(t[1] for t in traces)
    for size in sorted({t[0] for t in traces}):
        selected = [t for t in traces if t[0]==size and t[1]==longest]
        errors = np.array([[float(r['controller_relative_error']) for r in t[3]] for t in selected])
        x = 4*selected[0][2]*np.arange(1, longest+1)
        mean = errors.mean(axis=0)
        ax.plot(x, mean, label=f'{size} states, {len(selected)} seeds')
        ax.fill_between(x, errors.min(axis=0), errors.max(axis=0), alpha=.15)
    ax.set(xlabel='Perturbed equilibrations during calibration',
           ylabel='Relative error of four controller gains', yscale='log')
    ax.grid(alpha=.2)
    ax.legend()
    fig.savefig(args.output/'calibration.png', dpi=170)
    fig.savefig(args.output/'calibration.svg')
    write_json(args.output/'verified.json', dict(cases=rows,
        sources=[str(p.resolve()) for p in args.runs],
        task_training_performed=False,
        verification='coverage, known wiring, saved controller, true error, budgets and trajectory endpoint'))
    lines = ['# Exchanged-response controller calibration', '',
             'Exploratory calibration and saved gradient audits; no task training in these runs.', '',
             '| States | Updates | Perturbed equilibrations | Gain error, mean | Minimum gradient cosine |',
             '|---:|---:|---:|---:|---:|']
    for size, steps in sorted({(r['size'],r['steps']) for r in rows}):
        group = [r for r in rows if r['size']==size and r['steps']==steps]
        lines.append(f"| {size} | {steps} | {group[0]['perturbed_equilibrations']} | "
            f"{np.mean([r['controller_relative_error'] for r in group]):.6f} | "
            f"{min(r['gradient_cosine'] for r in group):.8f} |")
    lines += ['', 'Each case additionally needs one free anchor and eight anchor projection reads. '
        'Each perturbed equilibrium supplies one measured scalar. Four unknown gains and known '
        'wiring are essential assumptions; the fixed wiring contains 8n coefficients. No process '
        'noise is injected. Exact gradients are audit references only. Settling time and hardware '
        'overhead are not inferred from equilibrium counts.', '',
        'Plot lines are seed means; shaded bands show the minimum and maximum across seeds.',
        '', '![Calibration](calibration.png)', '']
    (args.output/'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
