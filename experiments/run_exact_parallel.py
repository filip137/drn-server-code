"""Small bounded transport for independent exact configs; no scientific changes."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from datetime import datetime, timezone


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('configs', nargs='+', type=Path)
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--output-root', required=True, type=Path)
    ap.add_argument('--dataset-root', required=True, type=Path)
    ap.add_argument('--target', required=True)
    ap.add_argument('--workers', type=int, required=True)
    ap.add_argument('--timeout-seconds', type=int, default=21570,
                    help='Operational time cap per worker; does not change configured epochs.')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if args.workers < 1 or args.timeout_seconds < 1:
        ap.error('workers and timeout-seconds must be positive')
    source = args.source.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / 'launcher-lock').mkdir()
    (output / 'launcher.pid').write_text(str(os.getpid()))
    stamp = lambda: datetime.now(timezone.utc).isoformat()
    (output / 'started_at').write_text(stamp())
    env = os.environ.copy()
    env.update(KMP_DISABLE_SHM='1', KMP_SHM_DISABLE='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', TF_CPP_MIN_LOG_LEVEL='2',
               MPLCONFIGDIR='/tmp/mpl-refined-conv3', GIT_CEILING_DIRECTORIES=str(source.parent),
               EXPERIMENT_SOURCE_COMMIT=(source / 'SOURCE_COMMIT').read_text().strip(),
               EXPERIMENT_SOURCE_ARCHIVE_SHA256=(source / 'SOURCE_ARCHIVE_SHA256').read_text().strip())
    subprocess.run(['sha256sum', '--check', '--quiet', 'LAYERWISE_SHA256SUMS'], cwd=source, check=True)
    import torch
    gpu = torch.cuda.get_device_properties(0)
    (output / 'gpu.json').write_text(json.dumps(dict(
        name=gpu.name, total_memory=gpu.total_memory, torch=torch.__version__,
        cuda=torch.version.cuda), indent=2) + '\n')
    stop = threading.Event()

    def run(config):
        if stop.is_set():
            return dict(config=str(config), state='not_started_after_operational_failure')
        config = config.resolve()
        case = config.stem
        cmd = ['timeout', '--signal=TERM', '--kill-after=20s', f'{args.timeout_seconds}s',
               sys.executable, '-m', 'experiments.exact_run', str(config),
               '--output-root', str(output / 'runs'), '--device', 'cuda',
               '--dataset-root', str(args.dataset_root), '--target', args.target,
               '--gradient-trace-samples-per-epoch', '8',
               '--summary-json', str(output / f'{case}.summary.json')]
        if args.smoke:
            cmd.append('--smoke')
        (output / f'{case}.command.json').write_text(json.dumps(cmd, indent=2) + '\n')
        (output / f'{case}.started_at').write_text(stamp())
        with (output / f'{case}.log').open('x') as log:
            p = subprocess.Popen(cmd, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            (output / f'{case}.pid').write_text(str(p.pid))
            code = p.wait()
        (output / f'{case}.exit_code').write_text(str(code))
        (output / f'{case}.finished_at').write_text(stamp())
        text = (output / f'{case}.log').read_text()
        scientific = code != 0 and 'NonFiniteTrainingError' in text and not args.smoke
        if code and not scientific:
            stop.set()
        result = dict(case=case, returncode=code,
                      state='complete' if code == 0 else 'scientific_nonfinite' if scientific else 'operational_failure')
        print(json.dumps(result), flush=True)
        return result

    code = 1
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(run, args.configs))
        (output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
        code = int(stop.is_set())
    finally:
        (output / 'finished_at').write_text(stamp())
        (output / 'exit_code').write_text(str(code))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
