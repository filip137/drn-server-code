"""Collect a bounded retry into its original CIFAR stage and continue its policy."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


QUEUE = Path.home()/'.codex/skills/run-watch/scripts/gpu_queue.py'


def tree_identity(root):
    identity = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symlinks are not permitted in collected bundles: {path}')
        relative = str(path.relative_to(root))
        if path.is_dir():
            identity[relative] = None
        elif path.is_file():
            with path.open('rb') as stream:
                identity[relative] = hashlib.file_digest(stream, 'sha256').hexdigest()
        else:
            raise ValueError(f'Unsupported bundle entry: {path}')
    return identity


def copy_bundle(source, destination):
    """Install a validated tree atomically; identical repeated collection is a no-op."""
    source, destination = Path(source), Path(destination)
    if not source.is_dir() or source.is_symlink() or destination.is_symlink():
        raise ValueError('Bundle paths must be real directories')
    expected = tree_identity(source)
    if not expected:
        raise ValueError('Cannot collect an empty bundle')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and any(destination.iterdir()):
        if tree_identity(destination) != expected:
            raise ValueError(f'Canonical bundle differs; refusing overwrite: {destination}')
        return False
    temporary = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-collect-', dir=destination.parent))
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        if tree_identity(temporary) != expected:
            raise ValueError('Retry bundle changed during collection')
        if destination.exists():
            destination.rmdir()  # Only an empty canonical directory may be replaced.
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return True


def load_wrapper(root):
    source = root/'source'
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('cifar_retry_wrapper', source/'experiments/run_cifar_beta_stages.py')
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    wrapper.ROOT = root
    return wrapper


def collect(original_root, retry_root, case, stage):
    original_root, retry_root = Path(original_root).resolve(), Path(retry_root).resolve()
    if original_root == retry_root or not case or Path(case).name != case or case in ('.', '..'):
        raise ValueError('Separate study roots and a simple case ID are required')
    if stage not in ('1', '5', '10', '20', '30', '50'):
        raise ValueError('Unsupported stage')
    analysis = original_root/'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    with (analysis/f'{case}-{stage}-retry-collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        job_id = os.environ['RUN_WATCH_JOB_ID']
        command = [sys.executable, str(QUEUE)]
        if os.environ.get('RUN_WATCH_CONFIG'):
            command += ['--config', os.environ['RUN_WATCH_CONFIG']]
        status = json.loads(subprocess.check_output(command + ['status', job_id], text=True))
        target = status['attempts'][-1]['target']
        relative = Path('cells')/stage/case
        retry_cell = retry_root/relative
        if target != 'local':
            remote = status['job']['spec']['placements'][target]['run_dir']
            retry_cell.mkdir(parents=True, exist_ok=True)
            subprocess.run(['rsync', '-a', '-e', 'ssh -F /home/filip/.ssh/config -o BatchMode=yes',
                            target+':'+remote+'/', str(retry_cell)+'/'], check=True)
        wrapper = load_wrapper(retry_root)
        receipt = wrapper.validate(stage, case)
        if receipt.get('validated') is not True:
            raise ValueError('Retry collection did not validate')
        copy_bundle(retry_cell, original_root/relative)
        receipt = dict(receipt, target=target, job_id=job_id, retry_root=str(retry_root),
                       original_root=str(original_root), retry_bundle=str(retry_cell),
                       canonical_bundle=str(original_root/relative))
        (retry_root/'analysis').mkdir(parents=True, exist_ok=True)
        name = f'{case}-{stage}-collection.json'
        wrapper.reporting.atomic_write_json(retry_root/'analysis'/name, receipt)
        wrapper.reporting.atomic_write_json(analysis/name, receipt)
        wrapper.ROOT = original_root
        wrapper.advance()
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root', type=Path, required=True)
    parser.add_argument('--retry-root', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--stage', required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.original_root, args.retry_root, args.case, args.stage)))


if __name__ == '__main__':
    main()
