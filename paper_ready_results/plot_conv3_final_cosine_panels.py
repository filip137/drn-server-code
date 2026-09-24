"""Plot saved final-checkpoint cosine summaries; no new measurements."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
STEM = 'conv3_p90_final_noise_cosine_20260921'
PARAMETERS = ('ConvWeight_0', 'ConvWeight_1', 'ConvWeight_2', 'DenseWeight_0')
COLORS = ('#0072B2', '#E69F00', '#009E73', '#CC79A7')
SCHEMES = ('baseline', 'legacy', 'ours')
SIGMAS = [0, 1e-5, 3e-5, 1e-4, 3e-4, 5e-4, 1e-3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accuracy', action='store_true', help='Add final validation accuracies.')
    args = parser.parse_args()
    with (HERE / f'{STEM}_summary.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    accuracies = {}
    if args.accuracy:
        for filename in ('conv3_p90_read_noise_20260919.csv', 'conv3_p90_read_noise_1em3_20260920.csv'):
            with (HERE / filename).open() as stream:
                for record in csv.DictReader(stream):
                    if record['state'] == 'complete':
                        assert int(record['epochs_completed']) == 30
                        accuracies[record['bundle']] = record
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    nrows = 3 if args.accuracy else 2
    fig, axes = plt.subplots(nrows, 3, figsize=(14, 11.2 if args.accuracy else 8.4),
                             sharex=True, sharey='row')
    for column, scheme in enumerate(SCHEMES):
        beta = float(next(r['injected_beta'] for r in rows if r['scheme'] == scheme))
        axes[0, column].set_title(f'{scheme.capitalize()}  ·  β = {beta:.5g}', pad=12)
        for row_index, noisy in enumerate((True, False)):
            ax = axes[row_index, column]
            for parameter, color in zip(PARAMETERS, COLORS):
                points = sorted((r for r in rows if r['scheme'] == scheme
                                 and r['parameter'] == parameter
                                 and float(r['readout_sigma']) == (float(r['training_sigma']) if noisy else 0)
                                 and (float(r['training_sigma']) > 0 or r['training_gpu'] == 'RTX5090')),
                                key=lambda r: float(r['training_sigma']))
                assert len(points) == (7 if scheme == 'baseline' else 6)
                x = [float(r['training_sigma']) for r in points]
                y = [float(r['cosine_median']) for r in points]
                ax.plot(x, y, color=color, marker='o', ms=4.5, lw=2)
                ax.fill_between(x, [float(r['cosine_p10']) for r in points],
                                [float(r['cosine_p90']) for r in points], color=color, alpha=.13, linewidth=0)
            ax.axhline(.9, color='#777777', lw=.9, ls=(0, (4, 4)), zorder=0)
            ax.axhline(0, color='#777777', lw=.8, zorder=0)
            ax.set_xscale('symlog', linthresh=1e-5)
            ax.set_xlim(-1e-6, 1.25e-3)
            ax.set_ylim(-1, 1.06)
            ax.set_yticks([-1, -.5, 0, .5, .9, 1])
            ax.set_xticks(SIGMAS, ['0', '10⁻⁵', '3×10⁻⁵', '10⁻⁴', '3×10⁻⁴', '5×10⁻⁴', '10⁻³'], rotation=45, ha='right')
            ax.grid(alpha=.15)
            if scheme != 'baseline':
                ax.axvspan(8e-4, 1.25e-3, color='#eeeeee', zorder=-1)
                ax.text(.96, .06, '10⁻³: training failed\nNo final checkpoint', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=9, color='#555555')
            if row_index == nrows - 1:
                ax.set_xlabel('Training read-noise σ')
        if args.accuracy:
            ax = axes[2, column]
            # Join by the exact source bundle so accuracy and cosine use identical weights.
            selected = sorted((r for r in rows if r['scheme'] == scheme
                               and r['parameter'] == PARAMETERS[0]
                               and float(r['readout_sigma']) == 0
                               and (float(r['training_sigma']) > 0 or r['training_gpu'] == 'RTX5090')),
                              key=lambda r: float(r['training_sigma']))
            x, y = [], []
            for r in selected:
                record = accuracies[r['source_bundle']]
                assert record['scheme'] == scheme
                assert float(record['sigma']) == float(r['training_sigma'])
                x.append(float(r['training_sigma']))
                y.append(float(record['final_validation_pct']))
            ax.plot(x, y, color='#333333', marker='o', lw=2, ms=4.5)
            for sigma, accuracy in zip(x, y):
                ax.annotate(f'{accuracy:.2f}', (sigma, accuracy), xytext=(6 if sigma == 0 else 0, 9),
                            textcoords='offset points', ha='left' if sigma == 0 else 'center', fontsize=9)
            ax.set_ylim(95, 99.25)
            ax.set_yticks([95, 96, 97, 98, 99])
            ax.grid(alpha=.15)
            ax.set_xlabel('Training read-noise σ')
            ax.set_xticks(SIGMAS, ['0', '10⁻⁵', '3×10⁻⁵', '10⁻⁴', '3×10⁻⁴', '5×10⁻⁴', '10⁻³'], rotation=45, ha='right')
            if scheme != 'baseline':
                ax.axvspan(8e-4, 1.25e-3, color='#eeeeee', zorder=-1)
                ax.text(.96, .06, '10⁻³: training failed\nNo epoch-30 accuracy', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=9, color='#555555')
    axes[0, 0].set_ylabel('Noisy readout\nEP–BPTT cosine')
    axes[1, 0].set_ylabel('Clean readout at the same weights\nEP–BPTT cosine')
    if args.accuracy:
        axes[2, 0].set_ylabel('Final validation accuracy (%)\nEpoch 30 · seed 0')
    handles = [Line2D([0], [0], color=c, marker='o', lw=2, label=l)
               for c, l in zip(COLORS, ('Conv layer 1', 'Conv layer 2', 'Conv layer 3', 'Dense output'))]
    handles.append(Line2D([0], [0], color='#777777', lw=.9, ls=(0, (4, 4)), label='Cosine 0.90'))
    title = 'Conv3: gradient alignment and accuracy' if args.accuracy else 'Conv3: gradient alignment at the end of training'
    fig.suptitle(f'{title} (epoch 30)', fontsize=16, y=.99)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .945), ncol=5, frameon=False)
    fig.text(.5, .013,
             'Median and 10–90% range · 36 matched batches · 4 draws per noisy batch · p90 betas · T = K = 8\n'
             'Training used V100/A100/RTX 5090; all replays used RTX 3090. Zero-noise points use RTX 5090 controls.'
             + ('\nAccuracy: existing 5,000-example MNIST validation measurements; read noise affects training gradient readout.' if args.accuracy else ''),
             ha='center', fontsize=9, color='#555555')
    fig.subplots_adjust(top=.89 if args.accuracy else .86, bottom=.14 if args.accuracy else .16,
                        left=.085, right=.985, hspace=.16, wspace=.13)
    for suffix in ('png', 'pdf', 'jpg'):
        extra = '_with_accuracy' if args.accuracy else ''
        path = HERE / f'{STEM}_panels{extra}.{suffix}'
        options = {'pil_kwargs': {'quality': 95, 'subsampling': 0}} if suffix == 'jpg' else {}
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white', **options)
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
