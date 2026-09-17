"""Summarize the frozen common-test comparisons without selecting on test data."""
import csv
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
KINDS = ['p0', 'previous_one_epoch', 'tuned_within_one_epoch', 'constant', 'exponential']
LABELS = ['Deployment', 'Previous\n1 epoch', 'Tuned\n≤1 epoch', 'Longer\nconstant', 'Decaying\nrate']


def read(path):
    return json.loads(Path(path).read_text())


def main():
    out = HERE / 'analysis'
    out.mkdir(exist_ok=True)
    rows = []
    complete = []
    for architecture, folder in [('drn', 'drn_readouts'), ('crossbar', 'crossbar_readouts_collected')]:
        root = HERE / folder
        if not (root / 'index.json').exists():
            continue
        index = read(root / 'index.json')
        assert index['status'] == 'complete' and len(index['entries']) == 30 and not index['test_selection']
        complete.append(architecture)
        for entry in index['entries']:
            replay = read(root / 'test' / Path(entry['test_readout']).name)
            assert replay['status'] == 'complete' and replay['writes'] == 0 and replay['unchanged_state_check']['passed']
            primary_key = 'apparent' if architecture == 'drn' else 'apparent_forward'
            secondary_key = 'persistent_secondary_diagnostic' if architecture == 'drn' else 'persistent_forward'
            evaluation = replay['evaluation']
            primary = evaluation[primary_key]
            # Readout reports inherit the original architecture's diagnostic key.
            if secondary_key not in evaluation:
                matches = [k for k, v in evaluation.items() if 'persistent' in k and isinstance(v, dict) and 'kl_teacher_student' in v]
                assert len(matches) == 1, evaluation.keys()
                secondary_key = matches[0]
            secondary = evaluation[secondary_key]
            assert primary['examples'] == secondary['examples'] == 10000
            development = entry['replica'] == ('array-1-write-1' if architecture == 'drn' else 'array-2090402')
            validation = entry.get('validation', {}).get(primary_key, {})
            epoch = entry.get('selected_epoch', replay.get('selected_epoch', 0 if entry['kind'] == 'p0' else None))
            pulses = entry.get('selected_pulses', 0 if entry['kind'] == 'p0' else None)
            if architecture == 'drn' and entry['kind'] == 'previous_one_epoch':
                legacy = read(Path(entry['state_path']).parents[1] / 'result.json')['metrics']
                validation = legacy['selected_apparent_validation']
                pulses = legacy['selected_applied_pulses']
                old_test = legacy['selected_test']['apparent']
                assert old_test['prediction_sha256'] == primary['prediction_sha256']
                assert abs(old_test['kl_teacher_student'] - primary['kl_teacher_student']) < 1e-7
            rows.append(dict(architecture=architecture, condition=entry['condition'], replica=entry['replica'],
                             cohort='development_array' if development else 'additional_array', method=entry['kind'],
                             initial_learning_rate=entry.get('learning_rate'), selected_epoch=epoch, selected_pulses=pulses,
                             validation_kl=validation.get('kl_teacher_student'),
                             test_kl=primary['kl_teacher_student'], test_accuracy=primary['student_accuracy'],
                             teacher_test_accuracy=primary['teacher_accuracy'], test_teacher_agreement=primary['teacher_agreement'],
                             persistent_diagnostic_test_kl=secondary['kl_teacher_student'],
                             persistent_diagnostic_test_accuracy=secondary['student_accuracy'],
                             readout=str(root / 'test' / Path(entry['test_readout']).name),
                             state_sha256=replay['state_sha256']))
    if not rows:
        print('Waiting for complete test readouts.')
        return
    with (out / 'paired_test_readouts.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    aggregates = []
    for architecture in complete:
        for condition in ['healthy', 'faulted']:
            for cohort in ['development_array', 'additional_array']:
                for method in KINDS:
                    group = [r for r in rows if (r['architecture'], r['condition'], r['cohort'], r['method']) == (architecture, condition, cohort, method)]
                    aggregates.append(dict(architecture=architecture, condition=condition, cohort=cohort, method=method,
                                           arrays=len(group), mean_test_kl=statistics.mean(r['test_kl'] for r in group),
                                           minimum_test_kl=min(r['test_kl'] for r in group), maximum_test_kl=max(r['test_kl'] for r in group),
                                           mean_test_accuracy=statistics.mean(r['test_accuracy'] for r in group),
                                           selected_epochs=[r['selected_epoch'] for r in group]))
    payload = dict(status='complete' if len(complete) == 2 else 'partial', completed_architectures=complete,
                   selection_used_test=False, rows=rows, aggregates=aggregates)
    (out / 'paired_test_readouts.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    lines = ['# Common-test recovery comparison', '',
             'Exploratory results. Rates and candidate epochs were selected using validation KL before these 10,000-example test replays. The table reports the mean across two additional saved arrays; development-array results and each individual array are in the CSV.', '',
             '| Architecture / condition | Deployment | Previous 1 epoch | Tuned ≤1 epoch | Longer constant | Decay |',
             '|---|---:|---:|---:|---:|---:|']
    for architecture in complete:
        for condition in ['healthy', 'faulted']:
            subset = [r for r in aggregates if r['architecture'] == architecture and r['condition'] == condition and r['cohort'] == 'additional_array']
            values = [next(r['mean_test_kl'] for r in subset if r['method'] == method) for method in KINDS]
            lines.append('| ' + architecture.upper() + ' / ' + condition + ' | ' + ' | '.join(f'{v:.5f}' for v in values) + ' |')
    lines += ['', 'Held-apparent teacher KL is primary; lower is better. Accuracy, teacher agreement, selected epoch, update pulses, and persistent-state diagnostics are in `paired_test_readouts.csv`. Both long-schedule columns retain their validation-selected checkpoint, including P0; they do not necessarily use epoch 30.', '',
              'These measurements establish the best results within the declared sweep, not a global optimum over all recovery methods. Two additional arrays provide a limited transfer check. DRNs and crossbars use different source teachers and device topologies, so their absolute KL values are not a matched architecture ranking.', '',
              'The DRN exact repeats retain the legacy replica-specific 55k/5k split. Some recovery validation examples were in the original teacher/HWA training split. The common official test cohort used here is separate and was not used to select the new rates or checkpoints.']
    (out / 'TEST_COMPARISON.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:10]))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(complete), 2, figsize=(11, 3.6 * len(complete)), squeeze=False, layout='constrained')
    for ai, architecture in enumerate(complete):
        for ci, condition in enumerate(['healthy', 'faulted']):
            ax = axes[ai, ci]
            subset = [r for r in rows if r['architecture'] == architecture and r['condition'] == condition and r['cohort'] == 'additional_array']
            for i, replica in enumerate(sorted({r['replica'] for r in subset})):
                values = [next(r['test_kl'] for r in subset if r['method'] == method and r['replica'] == replica) for method in KINDS]
                ax.plot(range(5), values, '-o', label=replica, color=['#0077b6', '#d67822'][i], alpha=.9, linewidth=1.5)
            ax.set_xticks(range(5), LABELS, fontsize=9)
            ax.set_ylabel('Held-apparent test KL')
            ax.set_title(architecture.upper() + ' — ' + condition)
            ax.grid(axis='y', alpha=.2)
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle('Frozen recovery choices on two additional arrays\nSame official 10,000-example test cohort; no test-based selection', fontsize=12)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(out / ('paired_test_comparison.' + extension), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
