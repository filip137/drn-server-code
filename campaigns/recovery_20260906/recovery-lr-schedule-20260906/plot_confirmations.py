"""Plot complete validation trajectories for the frozen additional-array checks."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def main():
    rows = []
    for architecture, folder in [('drn', 'drn_readouts'), ('crossbar', 'crossbar_readouts_collected')]:
        index_path = HERE / folder / 'index.json'
        if not index_path.exists():
            continue
        for entry in read(index_path)['entries']:
            if entry['kind'] not in {'constant', 'exponential'}:
                continue
            if entry['replica'] in {'array-1-write-1', 'array-2090402'}:
                continue
            if architecture == 'drn':
                result = read(entry['result_path'])
                curve = [dict(epoch=v['epoch'], kl=v['validation']['apparent']['kl_teacher_student'], pulses=v['pulses']) for v in result['candidates']]
                total_epochs = result['epochs_completed']
            else:
                suffix = entry['result_path'].split('/crossbar_confirmation/', 1)[1]
                result = read(HERE / 'crossbar_confirmation_collected' / suffix)['metrics']
                curve = [dict(epoch=0, kl=result['initial']['validation']['apparent_forward']['kl_teacher_student'], pulses=0)]
                curve += [dict(epoch=v['epoch'], kl=v['validation']['apparent_forward']['kl_teacher_student'], pulses=v['optimizer_cumulative']['applied_pulses']) for v in result['epochs']]
                total_epochs = result['training_final_epoch']
            best = next(v for v in curve if v['epoch'] == entry['selected_epoch'])
            rows.append(dict(architecture=architecture, condition=entry['condition'], replica=entry['replica'],
                             schedule=entry['kind'], starting_rate=entry['learning_rate'], total_epochs=total_epochs,
                             selected_epoch=best['epoch'], selected_validation_kl=best['kl'], selected_pulses=best['pulses'],
                             final_validation_kl=curve[-1]['kl'], total_pulses=curve[-1]['pulses'], trajectory=curve))
    if not rows:
        return
    out = HERE / 'analysis'
    fields = [k for k in rows[0] if k != 'trajectory']
    with (out / 'confirmation_trajectories.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows({k: row[k] for k in fields} for row in rows)
    (out / 'confirmation_trajectories.json').write_text(json.dumps(rows, indent=2) + '\n')
    architectures = [v for v in ['drn', 'crossbar'] if any(r['architecture'] == v for r in rows)]
    fig, axes = plt.subplots(len(architectures), 2, figsize=(11, 3.7 * len(architectures)), squeeze=False, layout='constrained')
    for ai, architecture in enumerate(architectures):
        for ci, condition in enumerate(['healthy', 'faulted']):
            ax = axes[ai, ci]
            subset = [r for r in rows if r['architecture'] == architecture and r['condition'] == condition]
            replicas = sorted({r['replica'] for r in subset})
            for row in subset:
                color = ['#0077b6', '#d67822'][replicas.index(row['replica'])]
                decay = row['schedule'] == 'exponential'
                curve = row['trajectory']
                ax.plot([v['epoch'] for v in curve], [v['kl'] for v in curve], color=color, linestyle='--' if decay else '-',
                        label=row['replica'] + (' decay' if decay else ' constant'), linewidth=1.8)
                ax.scatter(row['selected_epoch'], row['selected_validation_kl'], color=color, s=25,
                           marker='s' if decay else 'o', zorder=5)
            ax.set_title(architecture.upper() + ' — ' + condition)
            ax.set_xlabel('Recovery epoch')
            ax.set_ylabel('Held-apparent validation KL (log scale)')
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 3, 4, 6, 8)))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f'{value:g}'))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax.grid(alpha=.2, which='both')
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Frozen rates on two additional arrays\nMarkers indicate validation-selected states; final-epoch behavior remains visible', fontsize=12)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(out / ('confirmation_kl_vs_epochs.' + extension), dpi=180)
    plt.close(fig)
    for row in rows:
        print(row['architecture'], row['condition'], row['replica'], row['schedule'],
              f"best={row['selected_validation_kl']:.6f}@{row['selected_epoch']:.3g}", f"final={row['final_validation_kl']:.6f}")


if __name__ == '__main__':
    main()
