"""Direct overnight queue for the September 22 legacy/ours beta sweep.

Science lives in the six JSON configs; this script only handles transport,
idle-GPU admission, deadlines, progress collection, and bundle validation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time


STUDY = 'eqprop-conv3-beta-sweep-5em4-10ep-20260922-v2'
START = datetime.fromisoformat('2026-09-22T22:00:00+02:00').timestamp()
DEADLINE = datetime.fromisoformat('2026-09-23T07:55:00+02:00').timestamp()
CASE_SECONDS = 10800
PYTHON = '/home/filip/miniconda3/envs/py312/bin/python'
SSH = ['ssh', '-F', '/home/filip/.ssh/config', '-o', 'BatchMode=yes',
       '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=30',
       '-o', 'ServerAliveCountMax=3']
HOSTS = ['fifi', 'local', 'nom-cool-1', 'trex', 'loulou', 'riri']
PRIORITY = [.1, .3, .7]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def idle_snapshot(gpus, processes):
    """Allow an idle MPS server, never an active client or an occupied GPU."""
    rows = [line.split(',') for line in gpus.strip().splitlines() if line.strip()]
    if len(rows) != 1:
        return False
    name, memory, util = (item.strip() for item in rows[0])
    if not any(model in name for model in ('RTX 3090', 'RTX 5090')):
        return False
    if float(memory) > 700 or float(util) > 5:
        return False
    for line in processes.strip().splitlines():
        parts = line.split(',')
        if len(parts) < 2 or Path(parts[1].strip()).name != 'nvidia-cuda-mps-server':
            return False
    return True


def gpu_snapshot():
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    procs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                     '--format=csv,noheader,nounits'], text=True)
    # Existing serial queues retain their lane during gaps between workers.
    listing = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True)
    launchers = []
    for line in listing.splitlines():
        if STUDY in line or 'server_code' not in line:
            continue
        if re.search(r'/(?:launch[^ /]*\.sh|run_[^ /]*\.sh|[^ /]*queue[^ /]*\.py)(?:\s|$)', line) or 'experiments.run_cifar_lr_lane' in line:
            launchers.append(line.strip())
    return dict(at=stamp(), gpu=gpu, processes=procs, queue_launchers=launchers,
                idle=idle_snapshot(gpu, procs) and not launchers)


def progress(root, config_name):
    out = root / 'production' / Path(config_name).stem
    receipt = json.loads((out / 'case_status.json').read_text())
    info = dict(at=stamp(), receipt=receipt, gpu=gpu_snapshot(), alerts=[])
    files = list((out / 'runs').rglob('metrics.jsonl'))
    info['metrics'] = [dict(path=str(p), bytes=p.stat().st_size,
                            age_seconds=time.time() - p.stat().st_mtime) for p in files]
    log = out / 'training.log'
    if log.exists():
        stat = log.stat()
        info['log_bytes'] = stat.st_size
        info['log_age_seconds'] = time.time() - stat.st_mtime
        with log.open('rb') as stream:
            stream.seek(max(0, stat.st_size - 2500))
            info['log_tail'] = stream.read().decode(errors='replace')
    if receipt['state'] == 'running':
        started = datetime.fromisoformat(receipt['started_at']).timestamp()
        if time.time() - started > 300 and not list((out / 'runs').rglob('manifest.json')):
            info['alerts'].append('No semantic manifest five minutes after launch')
        if files and all(item['age_seconds'] > 2700 for item in info['metrics']):
            info['alerts'].append('Metrics have not advanced for 45 minutes')
        pid = receipt.get('worker_pid')
        if pid and not Path(f'/proc/{pid}').exists():
            info['alerts'].append('Recorded worker PID is missing')
    return info


def case_seconds(host):
    return CASE_SECONDS if host in ('local', 'nom-cool-1') else 7200


def fits(now, host='local', cases=1):
    return START <= now and now + cases * case_seconds(host) + 60 <= DEADLINE


def ordered_cases(rows):
    expected = [(beta, scheme) for beta in PRIORITY for scheme in ('legacy', 'ours')]
    assert len(rows) == len(expected)
    assert {(row['injected_beta'], row['scheme']) for row in rows} == set(expected)
    return [next(row for row in rows if (row['injected_beta'], row['scheme']) == key)
            for key in expected]


def choose_case(pending, assignments, host, now):
    """Reserve both same-beta cases on one host; admit a whole pair's budget."""
    for index, row in enumerate(pending):
        if assignments.get(str(row['injected_beta'])) == host and fits(now, host):
            return pending.pop(index)
    for index, row in enumerate(pending):
        beta = str(row['injected_beta'])
        if beta not in assignments and fits(now, host, cases=2):
            assignments[beta] = host
            return pending.pop(index)
    return None


def check_config(data):
    assert data['lab']['epochs'] == 10
    assert data['max_batches'] is None and data['max_test_batches'] is None
    assert data['evaluation']['official_test']['policy'] == 'disabled'
    assert data['eqprop']['endpoint_read_noise_std'] == .0005
    assert data['eqprop']['injected_beta_B'] in PRIORITY
    scale = {'legacy': 4096.0, 'ours': 64.0}[data['stability_pilot']['scheme']]
    assert data['eqprop']['amplification_factor'] == scale
    assert data['beta'] == data['eqprop']['injected_beta_B'] / scale


def validate_collected(source, directory):
    sys.path.insert(0, str(source))
    from experiments.reporting import validate_run
    results = list((directory / 'runs').rglob('result.json'))
    errors = []
    if len(results) != 1:
        errors.append(f'Expected one terminal result, found {len(results)}')
    for result in results:
        errors.extend(validate_run(result.parent))
        data = json.loads(result.read_text())
        if data['terminal_metrics'].get('official_test_read', False):
            errors.append('Unexpected official-test access')
    rows = json.loads((directory / 'summary.json').read_text())
    if len(rows) != 1 or rows[0]['epochs'] != 10:
        errors.append('Expected ten completed epochs')
    return errors


def case(root, config_name, host, *, smoke=False):
    source = root / 'source'
    out = root / ('gpu-smoke' if smoke else 'production') / config_name.removesuffix('.json')
    out.mkdir(parents=True, exist_ok=False)
    state = dict(state='preflight', at=stamp(), host=host, pid=os.getpid(), config=config_name)
    write(out / 'case_status.json', state)
    try:
        for _ in range(2):
            snapshot = gpu_snapshot()
            state['gpu'] = snapshot
            if not snapshot['idle']:
                raise RuntimeError('GPU occupied; no training admitted')
            time.sleep(2)
        if not smoke and not fits(time.time(), host):
            raise RuntimeError('Full ten-epoch budget does not fit before deadline')
        subprocess.run(['sha256sum', '--check', '--quiet', 'LAYERWISE_SHA256SUMS'],
                       cwd=source, check=True)
        subprocess.run(['sha256sum', '--check', '--quiet', 'SHA256SUMS'],
                       cwd=root / 'configs', check=True)
        config = root / 'configs' / config_name
        data = json.loads(config.read_text())
        check_config(data)
        env = dict(os.environ, KMP_DISABLE_SHM='1', KMP_SHM_DISABLE='1',
                   OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONUNBUFFERED='1', MPLCONFIGDIR='/tmp/mpl-legacy-beta-sweep',
                   TF_CPP_MIN_LOG_LEVEL='2', GIT_CEILING_DIRECTORIES=str(root),
                   EXPERIMENT_SOURCE_COMMIT=(source / 'SOURCE_COMMIT').read_text().strip(),
                   EXPERIMENT_SOURCE_ARCHIVE_SHA256=(source / 'SOURCE_ARCHIVE_SHA256').read_text().strip())
        seconds = 600 if smoke else min(case_seconds(host) - 30, int(DEADLINE - time.time()) - 30)
        cmd = ['timeout', '--signal=TERM', '--kill-after=20s', f'{seconds}s', sys.executable,
               '-m', 'experiments.exact_run', str(config), '--output-root', str(out / 'runs'),
               '--device', 'cuda', '--dataset-root', '/home/filip/datasets/mnist',
               '--target', host, '--gradient-trace-samples-per-epoch', '8',
               '--summary-json', str(out / 'summary.json')]
        if smoke:
            cmd.append('--smoke')
        state.update(state='running', started_at=stamp(), command=cmd)
        write(out / 'command.json', cmd)
        with (out / 'training.log').open('x') as log:
            process = subprocess.Popen(cmd, cwd=source, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            state['worker_pid'] = process.pid
            write(out / 'case_status.json', state)
            while process.poll() is None:
                state['heartbeat_at'] = stamp()
                write(out / 'case_status.json', state)
                time.sleep(10)
        state['returncode'] = process.returncode
        if process.returncode:
            scientific = 'NonFiniteTrainingError' in (out / 'training.log').read_text()
            state['state'] = 'scientific_nonfinite' if scientific else 'operational_failure'
        else:
            # validate_run resolves artifact paths on the execution host.
            sys.path.insert(0, str(source))
            from experiments.reporting import validate_run
            results = list((out / 'runs').rglob('result.json'))
            errors = ['Expected one result'] if len(results) != 1 else list(validate_run(results[0].parent))
            rows = json.loads((out / 'summary.json').read_text())
            if len(rows) != 1 or rows[0]['epochs'] != (1 if smoke else 10):
                errors.append('Epoch coverage mismatch')
            state.update(state='complete' if not errors else 'validation_failed', validation_errors=errors)
    except Exception as exc:
        state.update(state='operational_failure', error=repr(exc))
    state['finished_at'] = stamp()
    write(out / 'case_status.json', state)
    return 0 if state['state'] == 'complete' else 1


def schedule(root):
    (root / 'scheduler-lock').mkdir()
    records = {host: dict(state='scheduled') for host in HOSTS}
    rows = json.loads((root / 'cases.json').read_text())
    pending = ordered_cases(rows)
    assignments = {}
    lock = threading.Lock()
    state = dict(state='scheduled', pid=os.getpid(), start_utc=stamp(),
                 not_before='2026-09-22T20:00:00Z', cutoff='2026-09-23T05:55:00Z',
                 hosts=records, pair_hosts=assignments,
                 pending=pending, cases=[])

    def save():
        state['heartbeat_at'] = stamp()
        write(root / 'schedule_status.json', state)

    def remote_root(host):
        return root if host == 'local' else Path('/home/filip/server_code/results') / STUDY

    def call(host, args, timeout=60):
        command = args if host == 'local' else SSH + [f'filip@{host}', shlex.join(args)]
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=timeout)

    def collect(host):
        if host == 'local':
            return
        destination = root / 'collected' / host
        destination.mkdir(parents=True, exist_ok=True)
        subprocess.run(['rsync', '-az', '-e', shlex.join(SSH),
                        f'filip@{host}:{remote_root(host)}/production/', str(destination) + '/'],
                       check=True, timeout=180)

    def worker(host):
        base = remote_root(host)
        script = str(base / 'schedule.py')
        while fits(time.time(), host):
            with lock:
                if not pending:
                    records[host]['state'] = 'queue_drained'
                    save()
                    return
            try:
                snap = json.loads(call(host, [PYTHON, script, 'gpu', '--root', str(base)]))
                with lock:
                    records[host].update(state='checking', gpu=snap)
                    save()
                if not snap['idle']:
                    with lock:
                        records[host]['state'] = 'occupied_waiting'
                        save()
                    time.sleep(60)
                    continue
                with lock:
                    if not pending or not fits(time.time(), host):
                        return
                    row = choose_case(pending, assignments, host, time.time())
                    if row is None:
                        records[host]['state'] = 'no_unassigned_pair_fits'
                        save()
                        return
                    row.update(host=host, state='launching', assigned_at=stamp())
                    state['cases'].append(row)
                    records[host].update(state='launching', case=row['config'])
                    save()
                out = base / 'production' / Path(row['config']).stem
                cmd = [PYTHON, script, 'case', '--root', str(base), '--config', row['config'], '--host', host]
                # nohup survives SSH/controller disconnection; the case owns its timeout.
                shell = 'nohup ' + shlex.join(cmd) + ' > ' + shlex.quote(str(base / (Path(row['config']).stem + '.launcher.log'))) + ' 2>&1 < /dev/null & echo $!'
                pid = call(host, ['bash', '-c', shell]).strip()
                with lock:
                    row.update(state='running', launcher_pid=pid, command=cmd, output=str(out))
                    records[host]['state'] = 'running'
                    save()
                last_collect = 0
                while time.time() < DEADLINE + 60:
                    time.sleep(30)
                    observed = json.loads(call(host, [PYTHON, script, 'progress', '--root', str(base),
                                                      '--config', row['config']]))
                    receipt = observed['receipt']
                    with lock:
                        row.update(state=receipt['state'], receipt=receipt, progress=observed)
                        save()
                    if time.time() - last_collect > 1800 or receipt['state'] not in ('running', 'preflight'):
                        collect(host)
                        last_collect = time.time()
                    if receipt['state'] not in ('running', 'preflight'):
                        if receipt['state'] == 'complete':
                            local = out if host == 'local' else root / 'collected' / host / out.name
                            errors = validate_collected(root / 'source', local)
                            write(local / 'local-validation.json', dict(at=stamp(), errors=errors))
                            with lock:
                                row['local_validation_errors'] = errors
                                if errors:
                                    row['state'] = 'validation_failed'
                                save()
                        break
                with lock:
                    records[host]['state'] = 'case_finished'
                    save()
                if row['state'] in ('operational_failure', 'validation_failed', 'running', 'preflight'):
                    return
            except Exception as exc:
                with lock:
                    records[host].update(state='operational_failure', error=repr(exc))
                    save()
                return
        with lock:
            records[host]['state'] = 'insufficient_time_for_full_run'
            save()

    save()
    while time.time() < START:
        time.sleep(min(30, START - time.time()))
        save()
    with lock:
        state['state'] = 'running'
        save()
    with ThreadPoolExecutor(max_workers=len(HOSTS)) as pool:
        list(pool.map(worker, HOSTS))
    state['state'] = 'terminal_pending_review'
    state['finished_at'] = stamp()
    save()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['schedule', 'case', 'smoke', 'gpu', 'progress'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config')
    parser.add_argument('--host', default='local')
    args = parser.parse_args()
    if args.mode == 'gpu':
        print(json.dumps(gpu_snapshot()))
        return 0
    if args.mode == 'progress':
        print(json.dumps(progress(args.root.resolve(), args.config)))
        return 0
    if args.mode == 'schedule':
        return schedule(args.root.resolve())
    return case(args.root.resolve(), args.config, args.host, smoke=args.mode == 'smoke')


if __name__ == '__main__':
    raise SystemExit(main())
