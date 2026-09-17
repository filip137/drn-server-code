"""Validate matched arms and render the requested recovery figures."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'analysis'
OUT.mkdir(exist_ok=True)
results = {}
checks = []
for architecture in ('drn', 'crossbar'):
    for condition in ('healthy', 'faulted'):
        pair = []
        for writer in ('open_loop', 'closed_loop_pv'):
            path = ROOT / 'runs' / f'{architecture}-{condition}-{writer}' / 'result.json'
            result = json.loads(path.read_text())
            assert result['status'] == 'complete' and result['selected_replay_passed']
            assert len(result['epochs']) == 3
            assert all(e['examples'] == 55000 and e['batches'] == 3438 for e in result['epochs'])
            assert all(e['invariants']['bounds_respected'] and e['invariants']['faults_immutable'] for e in result['epochs'])
            assert result['selected_test']['apparent']['examples'] == 10000
            results[architecture, condition, writer] = result
            pair.append(result)
        left, right = pair
        for field in ('initial_state', 'inputs', 'seeds', 'learning_rate', 'epochs', 'objective', 'code_sha256', 'plan_sha256'):
            assert left['manifest'][field] == right['manifest'][field], (architecture, condition, field)
        assert left['initial'] == right['initial']
        assert left['initial_test'] == right['initial_test']
        assert [e['data_stream_sha256'] for e in left['epochs']] == [e['data_stream_sha256'] for e in right['epochs']]
        checks.append({'architecture': architecture, 'condition': condition, 'exact_P0_and_data_order': True, 'source_and_LR_matched': True, 'bounds_and_faults_verified': True})

colors = {'open_loop': '#be5c2b', 'closed_loop_pv': '#246b99'}
names = {'open_loop': 'Open-loop pulse', 'closed_loop_pv': 'Closed-loop P&V'}
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
columns = [('drn', 'healthy'), ('drn', 'faulted'), ('crossbar', 'healthy'), ('crossbar', 'faulted')]
fig, axes = plt.subplots(2, 4, figsize=(14, 6.8))
fig.subplots_adjust(left=.065, right=.985, bottom=.15, top=.86, hspace=.25, wspace=.24)
for col, (architecture, condition) in enumerate(columns):
    for writer in colors:
        r = results[architecture, condition, writer]
        values = [r['initial']] + [e['validation'] for e in r['epochs']]
        for row, (key, scale) in enumerate([('kl_teacher_student', 1), ('student_accuracy', 100)]):
            ax = axes[row, col]
            ax.plot(range(4), [v['apparent'][key]*scale for v in values], color=colors[writer], marker='o', ms=4, label=names[writer])
            ax.grid(alpha=.18)
            ax.set_xticks(range(4))
            if row == 1:
                ax.set_xlabel('Recovery epoch')
    axes[0, col].set_title(f'{architecture.upper() if architecture == "drn" else "Crossbar"} · {condition}')
axes[0, 0].set_ylabel('Validation teacher KL ↓')
axes[1, 0].set_ylabel('Validation accuracy (%) ↑')
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle('Recovery from the same HWA + P&V state', fontsize=15, y=.975)
fig.text(.5, .018, 'Held apparent state · One saved array/write per architecture · Matched learning rate and data order within each pair\nDRN: synthetic RESET-stuck faults; crossbar: native effective-weight faults. Exploratory model-based result.', ha='center', fontsize=9)
for extension in ('png', 'svg', 'pdf'):
    fig.savefig(OUT / f'open_vs_closed_recovery.{extension}', dpi=200)
plt.close(fig)

rows = []
for key, r in results.items():
    architecture, condition, writer = key
    row = dict(architecture=architecture, condition=condition, writer=writer, learning_rate=r['manifest']['learning_rate'], selected_epoch=r['selected']['epoch'], total_pulses=r['total_pulses'], verify_reads=r['verify_reads'], runtime_seconds=r['total_seconds'])
    for stage in ('initial', 'selected', 'final'):
        evaluation = r[f'{stage}_test']
        row[f'{stage}_test_accuracy_percent'] = 100*evaluation['apparent']['student_accuracy']
        row[f'{stage}_test_KL'] = evaluation['apparent']['kl_teacher_student']
        row[f'{stage}_persistent_test_accuracy_percent'] = 100*evaluation['persistent_secondary_diagnostic']['student_accuracy']
        row[f'{stage}_persistent_test_KL'] = evaluation['persistent_secondary_diagnostic']['kl_teacher_student']
    row['fault_pulses'] = r['epochs'][-1]['invariants']['fault_pulses']
    rows.append(row)
with (OUT / 'comparison.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
fig.subplots_adjust(left=.09, right=.98, bottom=.16, top=.86, hspace=.27, wspace=.24)
for col, architecture in enumerate(('drn', 'crossbar')):
    for row, (metric, label) in enumerate([('KL', 'Test teacher KL ↓'), ('accuracy_percent', 'Test accuracy (%) ↑')]):
        ax = axes[row, col]
        for x, condition in enumerate(('healthy', 'faulted')):
            matched = [r for r in rows if r['architecture'] == architecture and r['condition'] == condition]
            initial = matched[0][f'initial_test_{metric}']
            ax.plot(x-.2, initial, 's', color='#777777', label='HWA + P&V (P0)' if x == 0 else None)
            for offset, writer in zip((0, .2), colors):
                r = next(r for r in matched if r['writer'] == writer)
                ax.plot(x+offset, r[f'selected_test_{metric}'], 'o' if writer == 'open_loop' else 'D', color=colors[writer], label=names[writer] if x == 0 else None)
        ax.set_xticks([0, 1], ['Healthy', 'Corrupt devices'])
        ax.set_xlim(-.45, 1.45)
        ax.grid(axis='y', alpha=.18)
        ax.set_ylabel(label)
    axes[0, col].set_title('DRN' if architecture == 'drn' else 'Crossbar')
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle('Open-loop versus closed-loop recovery: selected test states', fontsize=13, y=.975)
fig.text(.5, .015, 'Selection: minimum validation KL among epochs 0–3 · Held apparent state · One array/write per architecture\nWithin-architecture writer comparison; teachers and fault definitions differ across architectures.', ha='center', fontsize=8)
for extension in ('png', 'svg', 'pdf'):
    fig.savefig(OUT / f'open_vs_closed_test.{extension}', dpi=200)
plt.close(fig)

(OUT / 'verification.json').write_text(json.dumps({'status': 'complete', 'pairs': checks, 'arms': len(rows)}, indent=2)+'\n')
lines = ['# Open-loop versus closed-loop HWA/P&V recovery', '', 'Completed exploratory comparison: eight arms, three full epochs each, one saved array/write per architecture. Selection uses minimum held-apparent validation teacher KL with P0 eligible; test is evaluated only after training and selection. All pairs reuse exact P0 tensors, source artifacts, seeds, learning rates, and training order.', '', '![Recovery trajectories](open_vs_closed_recovery.png)', '', '| Architecture | State | Writer | Selected epoch | Test accuracy: P0 → selected | Test KL: P0 → selected | Final test accuracy | Update pulses |', '|---|---|---|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['architecture']} | {r['condition']} | {names[r['writer']]} | {r['selected_epoch']} | {r['initial_test_accuracy_percent']:.2f}% → {r['selected_test_accuracy_percent']:.2f}% | {r['initial_test_KL']:.5f} → {r['selected_test_KL']:.5f} | {r['final_test_accuracy_percent']:.2f}% | {r['total_pulses']:,} |")
lines += ['', 'The closed-loop controller accumulates digital Adam targets and applies SET/RESET pulses until the held apparent observation is within the declared tolerance, with at most 128 pulses per cell per minibatch and 640 during recovery. It never uses the fault mask, hidden persistent state or per-cell bounds in a programming decision. The pulse plant enforces true bounds and immobile faults. Untouched cells keep their apparent state; verify reads do not redraw write noise.', '', 'DRN tolerance is 0.023725 in progress; crossbar tolerance is 0.04745 in q=a−r. DRN learning rate is 1e−4; crossbar rates are 1e−5 healthy and 3e−5 faulted, retained from the reviews and shared across writers. Digital target memory is additional controller state. This comparison matches optimizer/data budgets, and reports the extra write/verify work separately.', '', 'These are model-based exploratory results from one array/write, with different teachers and fault models across architectures. They isolate writer effects within each architecture and do not establish an architecture ranking or measured-device energy cost. The CSV includes persistent-state diagnostics and fixed-final results; the curve uses validation metrics, whereas the table uses the selected test endpoint.', '', '[Full metrics](comparison.csv) · [Pairing checks](verification.json) · [Study contract](../plan.json)']
interpretation = [
    'P&V gives the largest benefit in the DRN: selected test accuracy reaches **98.13% healthy and 98.08% faulted**, compared with 97.76% and 97.59% for open-loop selection. Test KL is about 75% lower than the corresponding selected open-loop value in both conditions. The corrupt crossbar reaches 96.82% versus 96.76%, with about 17% lower KL; six additional correct test classifications on one array/write should not be treated as a population-level accuracy advantage.',
    'The healthy crossbar makes **zero P&V writes** at the retained learning rate and tolerance. Its accumulated target movement stays inside the verify window for all three epochs. The open-loop healthy crossbar obtains a small selected improvement and then deteriorates. This is a result for the retained recovery settings, not an independently optimized closed-loop learning-rate study.',
    'The three P&V arms that write all select epoch three, the budget boundary. Their curves have not established convergence. The test figure shows validation-selected endpoints; the trajectory figure and CSV retain the fixed-final open-loop degradation.',
    'Corrupt devices remain persistently immobile. Attempted writes can select a more favorable held apparent noise realization at such devices. The result therefore combines compensation through healthy cells and apparent-noise conditioning; those contributions are not isolated here. At final P&V, 99.94%/99.92% of healthy/faulted DRN apparent states meet the target tolerance, versus 28.15%/27.69% for the corresponding persistent-state diagnostic. High apparent acceptance is not a persistent-programming correctness claim.',
    'Full pulse-resolved P&V was retained. A fused CUDA implementation was verified bitwise against the reference pulse plant and a 512-minibatch recovery trajectory before replacing the two slow DRN attempts. The reference attempts remain under `attempts/`, and all six previously completed results were preserved byte-for-byte. [Acceleration record](../operational_acceleration.json) · [Exact replay evidence](../smoke/full-controller-parity.json) · [Pulse/bound diagnostics](writer_diagnostics.json)',
    '![Validation-selected test endpoints](open_vs_closed_test.png)',
]
lines[2:2] = [''] + [value for paragraph in interpretation for value in (paragraph, '')]
(OUT / 'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(rows, indent=2))
