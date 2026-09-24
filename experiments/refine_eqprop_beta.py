"""Refine observed beta crossings using the unchanged checkpoint replay."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(p)


def grid(lo, hi, ratio):
    n = math.ceil(math.log(hi / lo) / math.log(ratio))
    return [float(f'{lo * (hi / lo) ** (i / n):.12g}') for i in range(1, n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    spec = read(root / 'study.json')
    sys.path.insert(0, str(root / 'runtime'))
    from experiments.reporting import validate_run
    for name, expected in spec['source_hashes'].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    arch = spec['architecture']
    start = time.monotonic()
    state = dict(state='running', pid=os.getpid(), architecture=arch,
                 target=spec['target'], current_case=None, runs=[],
                 official_test_read=False, optimizer_steps_applied=False)
    progress = root / ('smoke_progress.json' if args.smoke else 'progress.json')
    assert not progress.exists(), 'Use a new attempt after reconciling existing progress.'
    observations = spec['old_measurements']
    observed = {scheme: {float(r['beta']): r for r in records}
                for scheme, records in observations.items()}

    def save():
        state['elapsed_seconds'] = time.monotonic() - start
        state['updated_at_unix'] = time.time()
        write(progress, state)

    def run(scheme, beta, smoke=False):
        name = f'{arch}_{scheme}_beta{beta:.12g}'.replace('.', 'p')
        state['current_case'] = name
        save()
        config = read(root / 'templates' / f'{scheme}.json')
        case = config['cases'][0]
        scale = case['injected_beta'] / case['base_beta']
        case.update(injected_beta=beta, base_beta=beta / scale)
        config_path = root / 'configs' / f'{name}.json'
        config_path.parent.mkdir(exist_ok=True)
        if config_path.exists():
            assert read(config_path) == config
        else:
            write(config_path, config)
        output = root / ('smoke' if smoke else 'production')
        command = [sys.executable, str(root / 'source/experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py'),
                   '--config', str(config_path), '--output-root', str(output),
                   '--run-id', name, '--device', 'cuda', '--target', spec['target']]
        if smoke:
            command.append('--smoke')
        logs = root / 'logs'
        logs.mkdir(exist_ok=True)
        log = logs / (('smoke_' if smoke else '') + name + '.log')
        remaining = spec.get('cap_seconds', 7200) - spec.get('smoke_budget_seconds', 120) - (time.monotonic() - start)
        if remaining < 180:
            raise TimeoutError('Recorded target budget exhausted.')
        print('START', name, flush=True)
        with log.open('x') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=min(900, remaining))
        bundle = output / name
        if result.returncode:
            message = log.read_text()
            st = read(bundle / 'status.json') if (bundle / 'status.json').exists() else {}
            if (not smoke and st.get('state') == 'failed' and 'out of memory' not in message.lower()
                    and any(x in message for x in ('NonFiniteTrainingError', 'contains non-finite', 'Non-finite', 'non-finite values'))):
                row = dict(beta=beta, state='numerical_failure', minimum_cosine=None, bundle=str(bundle))
            else:
                raise RuntimeError(f'Operational replay failure: {log}')
        else:
            assert not validate_run(bundle), bundle
            m = read(bundle / 'result.json')['terminal_metrics']
            count = 1 if smoke else 72
            assert m['replay_count'] == count and m['layer_comparison_count'] == count * (int(arch[-1])+1)
            assert not m['official_test_read'] and not m['optimizer_steps_applied']
            assert m['source_bytes_unchanged'] and m['all_bias_tensors_exact_zero']
            assert m['endpoint_read_noise_std'] == 0
            with (bundle / 'layer_metrics.csv').open() as stream:
                layers = list(csv.DictReader(stream))
            assert len(layers) == count * (int(arch[-1])+1)
            cosines = [float(r['cosine']) for r in layers]
            row = dict(beta=beta, state='complete', minimum_cosine=min(cosines) if all(map(math.isfinite, cosines)) else None,
                       maximum_norm_mismatch=max(float(r['symmetric_norm_delta']) for r in layers), bundle=str(bundle))
        row['scheme'] = scheme
        state['runs'].append(row)
        if not smoke:
            observed[scheme][beta] = row
        state['current_case'] = None
        save()
        print('DONE', len(state['runs']), name, row['state'], row['minimum_cosine'], flush=True)

    save()
    try:
        if args.smoke:
            run('baseline', 100., True)
        else:
            smoke = read(root / 'smoke_progress.json')
            assert smoke['state'] == 'complete'
            assert not validate_run(Path(smoke['runs'][0]['bundle']))
            for scheme in ('baseline', 'ours', 'legacy'):
                for beta in spec['initial_betas'][scheme]:
                    if beta not in observed[scheme]:
                        run(scheme, beta)
            refinement = {}
            for scheme in ('baseline', 'ours', 'legacy'):
                points = set()
                for threshold in (.90, .95):
                    passing = [b for b, r in observed[scheme].items()
                               if r.get('minimum_cosine') is not None and r['minimum_cosine'] > threshold]
                    lo = max(passing)
                    higher = sorted(b for b in observed[scheme] if b > lo)
                    if higher:
                        points.update(grid(lo, higher[0], 1.05))
                refinement[scheme] = sorted(points - observed[scheme].keys())
            assert len(state['runs']) + sum(map(len, refinement.values())) <= 75
            write(root / 'refinement_betas.json', refinement)
            for scheme, betas in refinement.items():
                for beta in betas:
                    run(scheme, beta)
            selections = []
            for scheme in ('baseline', 'ours', 'legacy'):
                for threshold in (.90, .95):
                    lo = max(b for b, r in observed[scheme].items()
                             if r.get('minimum_cosine') is not None and r['minimum_cosine'] > threshold)
                    higher = sorted(b for b in observed[scheme] if b > lo)
                    hi = higher[0] if higher else None
                    if hi:
                        assert hi / lo <= 1.050000001
                    selections.append(dict(scheme=scheme, threshold=threshold, beta=lo,
                        minimum_cosine=observed[scheme][lo]['minimum_cosine'], next_failing_beta=hi,
                        relative_bracket_width=hi/lo-1 if hi else None, open_upper_edge=hi is None))
            write(root / 'selections.json', selections)
        state['state'] = 'complete'
    except BaseException as error:
        state.update(state='failed', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
