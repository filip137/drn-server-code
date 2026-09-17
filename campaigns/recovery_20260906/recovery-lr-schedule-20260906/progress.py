"""Read concise, artifact-backed progress without touching experiment state."""
import datetime
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
now = time.time()
print(datetime.datetime.fromtimestamp(now, datetime.timezone.utc).isoformat(timespec='seconds'))
for name in ['drn_screen', 'drn_lr_boundary', 'drn_confirmation', 'drn_readouts']:
    path = ROOT / name / 'status.json'
    if not path.exists():
        print(name, 'not started')
        continue
    state = json.loads(path.read_text())
    if state['status'] == 'complete':
        completed = len(state['completed']) if 'completed' in state else 1
        print(name, 'complete', 'completed', completed)
        continue
    print(name, state['status'], 'completed', len(state.get('completed', [])),
          'heartbeat_age_s', round(now - state.get('heartbeat', now)),
          'pid', state.get('pid'))
    active = state.get('active', [])
    if state.get('child_pid'):
        active = active + [dict(case=state.get('current_case', state.get('arm')),
                               pid=state['child_pid'], result=state.get('result'))]
    for child in active:
        result = child.get('result')
        worker_path = Path(result).parent / 'status.json' if result else None
        worker = json.loads(worker_path.read_text()) if worker_path and worker_path.exists() else {}
        print(' ', child['case'], 'pid', child['pid'], 'epoch', worker.get('epoch'),
              'batch', worker.get('batch'), 'last_full_kl', worker.get('validation_kl'),
              'best_kl', worker.get('selected_kl'),
              'best_epoch', worker.get('selected_epoch'),
              'artifact_age_s', round(now - worker.get('heartbeat', now)))
    if state.get('failures') or state['status'] in {'failed', 'blocked_by_failed_screen', 'blocked_by_failed_confirmation'}:
        print('ATTENTION', json.dumps(state))
