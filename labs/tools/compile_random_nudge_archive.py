"""Archive the September 11--14 experiment evidence without rerunning experiments.

Only small reports/receipts are embedded. CSVs are indexed by hash and row count;
datasets, checkpoints and plots stay in simulation_results. Historical launcher
statuses are records, not process-liveness checks. The curated study statuses
distinguish cancelled grids from completed diagnostics on their checkpoints.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'docs/random_nudge_archive'
BASE = '9083aa2217ff244e816b562353e3bf937ee1c13e'
CUTOFF = '240b6637'


def artifacts(directory, *names):
    return [f'simulation_results/{directory}/{name}' for name in names]


CIRC = 'circulation_feedback_20260913'
DIRECTED = 'directed_eqprop_20260914'
FASHION = 'fmnist_homeostasis_20260914'
STUDIES = [
    ('E01', 'Initial Hopfield identities, finite nudges and teacher task', 'complete',
     ['docs/random_nudge_hopfield.md'] + artifacts('random_nudge_full', 'run.json', 'status.json', 'summary.json', 'report.md')),
    ('E02', 'Fixed-equilibrium estimator and response-design comparison', 'complete',
     ['docs/adjoint_measurement_methods.md'] + artifacts('adjoint_measurement_comparison', 'config.json', 'completion.json', 'measurements.csv')),
    ('E03', 'Recurrent digits training, LR screen and orthogonal-MC follow-up', 'complete',
     artifacts('digits_lr_calibration', 'summary.json') + artifacts('digits_combined', 'report.md', 'summary.json', 'verification.json')),
    ('E04', 'Synthetic and frozen-network probe scaling', 'complete',
     ['docs/random_nudge_scaling.md'] + artifacts('mc_scaling_diagnostics', 'run.json', 'status.json', 'statistical.csv', 'frozen_gradients.csv')),
    ('E05', 'Physical width scaling and matched read-noise control', 'complete',
     artifacts('mc_scaling_combined', 'report.md', 'summary.json', 'verification.json', 'noise_control_verification.json')),
    ('E06', 'Local/learned baselines and optimizer selection', 'complete',
     ['docs/random_nudge_improvements.md'] + artifacts('nudge_improvements_20260913/combined', 'report.md', 'summary.json', 'verification.json', 'held_out_baseline_audit.json')),
    ('E07', 'Frozen predictor audit and vector-fit algebra', 'complete_diagnostic',
     ['docs/direct_feedback_alignment_connection.md'] + artifacts('dfa_connection_20260913', 'frozen_predictor_audit.json')),
    ('E08', 'Dense stochastic circulation calibration and timestep refinement', 'complete',
     ['docs/circulation_feedback_theory.md', 'docs/circulation_feedback_experiments.md'] + artifacts(f'{CIRC}/main_verified', 'report.md', 'summary.json', 'verification.json')),
    ('E09', 'Four-loop stochastic calibration and same-network dense controls', 'complete',
     artifacts(f'{CIRC}/structured_verified', 'report.md', 'verification.json')),
    ('E10', 'Swapped DC response calibration at 10/60 updates', 'complete_calibration',
     ['docs/response_loop_calibration.md'] + artifacts(f'{CIRC}/response_verified', 'report.md', 'verified.json')),
    ('E11', 'Training with the frozen DC-calibrated controller', 'complete',
     artifacts(f'{CIRC}/response_training_verified', 'report.md', 'summary.json', 'verification.json')),
    ('E12', 'Structured-controller full-MNIST extension', 'cancelled_by_user',
     ['docs/mnist_feedback_experiments.md'] + artifacts('mnist_eqprop_20260914/main', 'run.json', 'cancellation.json')),
    ('E13', 'Untied-weight MNIST: all eight methods and focused comparison', 'complete',
     ['docs/directed_eqprop_mnist.md', 'docs/zero_order_vs_vf_homeostasis.md'] + artifacts(f'{DIRECTED}/main', 'report.md', 'completion.json', 'verification.json') + artifacts(f'{DIRECTED}/zero_order_vs_vf_homeostasis', 'report.md')),
    ('E14', 'Untied-MNIST finite-nudge, noise and probe-budget audit', 'complete_diagnostic',
     artifacts(f'{DIRECTED}/main', 'probe_budget_diagnostic.json')),
    ('E15', 'Shortened Fashion-MNIST homeostasis comparison', 'cancelled_by_user_partial_results',
     ['docs/fmnist_homeostasis_replication.md'] + artifacts(f'{FASHION}/main', 'run.json', 'cancellation.json') + artifacts(f'{FASHION}/smoke', 'completion.json')),
    ('E16', 'Public-author source and direct JAX/Torch parity', 'complete_diagnostic',
     ['docs/jacobian_homeostasis_connection.md'] + artifacts(FASHION, 'author_source.json', 'parity_toy.json', 'parity_trained.json')),
    ('E17', 'Fashion-MNIST settling and period-two audit', 'complete_diagnostic',
     artifacts(FASHION, 'settling_seed0.json')),
    ('E18', 'Fashion-MNIST probe bias and ideal MC variance', 'complete_diagnostic',
     artifacts(FASHION, 'probe_error_seed0.json')),
    ('E19', 'Frozen Fashion-MNIST feedback and parameter-gradient cosines', 'complete_diagnostic',
     ['docs/fmnist_feedback_cosines.md'] + artifacts(f'{FASHION}/cosines', 'run.json', 'completion.json', 'provenance.json', 'report.md', 'summary.json', 'feedback_cosines.csv', 'parameter_gradients.csv')),
    ('E20', 'Clamped-force mean-homeostasis identity', 'complete_algebra_check',
     ['docs/on_chip_asymmetry_direction.md'] + artifacts('on_chip_direction_20260914', 'clamped_homeostasis_identity.json')),
]


def snapshot(relative):
    path = ROOT / relative
    if not path.is_file():
        raise ValueError(f'Expected existing evidence file; got {relative}')
    raw = path.read_bytes()
    entry = dict(path=relative, sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    if path.suffix == '.json':
        entry.update(format='json', data=json.loads(raw))
    elif path.suffix == '.md':
        entry.update(format='markdown', text=raw.decode())
    elif path.suffix == '.csv':
        rows = csv.reader(io.StringIO(raw.decode()))
        entry.update(format='csv_index', columns=next(rows), rows=sum(1 for _ in rows))
    else:
        raise ValueError(f'Expected JSON, Markdown or CSV evidence; got {relative}')
    return entry


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def build():
    studies = []
    sources = {}
    for study_id, title, status, paths in STUDIES:
        studies.append(dict(id=study_id, title=title, status=status, sources=paths))
        for relative in paths:
            sources.setdefault(relative, snapshot(relative))

    # Preserve every actually completed Fashion-MNIST endpoint without treating
    # missing/partial seeds as completed runs or averaging unmatched seed sets.
    fashion = json.loads((ROOT / f'simulation_results/{FASHION}/main/cancellation.json').read_text())
    assert len(fashion['completed']) == 7 and len(fashion['partial']) == 4
    assert len(fashion['not_started']) == 9 and fashion['expected'] == 20
    endpoint_paths = artifacts(f'{FASHION}/main', *[f'{name}/result.json' for name in fashion['completed']])
    for relative in endpoint_paths:
        sources[relative] = snapshot(relative)
        assert sources[relative]['data']['status'] == 'complete'
    studies[14]['sources'].extend(endpoint_paths)

    def data(relative):
        return sources[f'simulation_results/{relative}']['data']

    # Reconcile the report's counts with archived verification/cancellation
    # receipts. This checks coverage records, not the numerical models again.
    assert data('random_nudge_full/status.json')['completed_cases'] == dict(ideal=72, finite=84, train=36)
    assert len(data('digits_combined/verification.json')['checkpoint_replays']) == 30
    assert len(data('mc_scaling_combined/verification.json')['checkpoints']) == 33
    assert data('nudge_improvements_20260913/combined/summary.json')['verified_checkpoints'] == 66
    assert data(f'{CIRC}/main_verified/verification.json')['coverage']['main_grid_complete']
    assert len(data(f'{CIRC}/main_verified/verification.json')['checkpoint_replays']) == 45
    assert data(f'{CIRC}/structured_verified/verification.json')['checkpoint_replays'] == 24
    assert len(data(f'{CIRC}/response_verified/verified.json')['cases']) == 12
    assert data(f'{CIRC}/response_training_verified/verification.json')['dc_checkpoint_replays'] == 6
    assert data(f'{DIRECTED}/main/completion.json')['coverage']['verified'] == 48
    cancelled = data('mnist_eqprop_20260914/main/cancellation.json')
    assert cancelled['completed_training'] == 0 and len(cancelled['partial_trajectories']) == 8
    cosine = data(f'{FASHION}/cosines/completion.json')
    assert cosine['checkpoints'] == 7 and cosine['selected_examples'] == 27
    assert cosine['checkpoint_hashes_unchanged'] and cosine['parameter_tensors_unchanged']

    # Pin historical launchers so later runs do not silently extend the archive.
    launcher_index = (OUTPUT / 'launcher_directories.json').read_bytes()
    launchers = []
    for relative in json.loads(launcher_index):
        directory = ROOT / relative
        records = [snapshot(f'{relative}/run.json')]
        for name in ('status.json', 'completion.json', 'completed.json', 'cancellation.json'):
            if (directory / name).is_file():
                records.append(snapshot(f'{relative}/{name}'))
        launchers.append(dict(directory=relative, receipts=records))

    history = git('log', '--reverse', '--format=%H\t%ad\t%s', '--date=short', f'{BASE}..{CUTOFF}')
    archive = dict(schema_version=1, date='2026-09-14', branch='codex/hopfield-random-nudge-adjoint',
        base_commit=git('rev-parse', BASE), experiment_history_through=git('rev-parse', CUTOFF),
        verification_scope='Source existence, JSON/CSV parsing, content hashes and saved coverage receipts; no new checkpoint replay or training.',
        counting='Studies share controls and checkpoints. Do not sum family counts as independent runs. Cancelled grids remain incomplete.',
        raw_artifacts='Datasets, checkpoints, full CSV rows and plots remain in ignored simulation_results; small evidence is embedded here.',
        launcher_status_scope='Historical saved statuses only; use cancellation/terminal receipts and curated study status, not a stale running heartbeat, to interpret coverage.',
        studies=studies, sources=list(sources.values()), launchers=launchers,
        launcher_index_sha256=hashlib.sha256(launcher_index).hexdigest(),
        commits=[dict(zip(('sha', 'date', 'subject'), line.split('\t', 2))) for line in history.splitlines()])
    return json.dumps(archive, indent=2, ensure_ascii=False, allow_nan=False) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Compare the committed snapshot to local sources without changing files')
    args = parser.parse_args()
    text = build()
    path = OUTPUT / 'evidence.json'
    if args.check:
        if not path.is_file() or path.read_text() != text:
            raise RuntimeError('Expected archive to match its declared local evidence; regenerate and review the difference')
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    data = json.loads(text)
    print(json.dumps(dict(status='verified' if args.check else 'compiled', studies=len(data['studies']),
        source_snapshots=len(data['sources']), launcher_receipts=len(data['launchers']),
        commits=len(data['commits']), bytes=len(text.encode()), path=str(path))))


if __name__ == '__main__':
    main()
