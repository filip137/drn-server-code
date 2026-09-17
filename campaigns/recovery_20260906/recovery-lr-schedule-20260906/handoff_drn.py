"""Let the active DRN arm finish, then replace only its serial supervisor."""
import json
import os
from pathlib import Path
import signal
import sys
import time

from parallel_runs import save

HERE = Path(__file__).resolve().parent
out = HERE / 'drn_screen'
request = json.loads((out / 'parallel_handoff_request.json').read_text())
parent_pid = request['supervisor_pid']
worker_pid = request['worker_pid']
result_path = Path(request['result_path'])


def live(pid):
    path = Path('/proc') / str(pid) / 'stat'
    if not path.exists():
        return False
    return path.read_text().split(') ', 1)[1].split()[0] != 'Z'


while not result_path.exists():
    if not live(worker_pid):
        raise RuntimeError('Current DRN worker ended before a complete result; inspect its log.')
    worker_status = json.loads((result_path.parent / 'status.json').read_text())
    if worker_status['status'] == 'failed':
        raise RuntimeError(worker_status)
    request.update(status='waiting_for_current_arm', heartbeat=time.time(), worker_epoch=worker_status.get('epoch'))
    save(out / 'parallel_handoff_request.json', request)
    time.sleep(10)
result = json.loads(result_path.read_text())
if result['status'] != 'complete' or result['epochs_completed'] != 30 or not result['selected_replay_passed']:
    raise RuntimeError('Expected a complete validated original arm before handoff.')
while live(worker_pid):
    time.sleep(1)
command = (Path('/proc') / str(parent_pid) / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
if 'run_screen.py drn' not in command:
    raise RuntimeError('The stopped supervisor no longer matches the recorded command.')
os.kill(parent_pid, signal.SIGTERM)
os.kill(parent_pid, signal.SIGCONT)
while live(parent_pid):
    time.sleep(1)
previous = json.loads((out / 'status.json').read_text())
previous.update(handoff_verified_no_live_worker=True, handoff_at=time.time(),
                reason='Subprocess concurrency only; numerical commands, inputs, seeds, epochs and completed runs unchanged.')
save(out / 'serial_supervisor_at_handoff.json', previous)
request.update(status='complete', heartbeat=time.time())
save(out / 'parallel_handoff_request.json', request)
os.execv(sys.executable, [sys.executable, str(HERE / 'parallel_runs.py'), '--workers', '3', '--mnist-root', '/home/filip/datasets/mnist'])
