"""Read actual heartbeat, metric and exit evidence without changing a run."""
from pathlib import Path
import json
import time

root = Path(__file__).resolve().parents[2]
now = time.time()
for path in sorted((root / 'runs').glob('*/status.json')):
    state = json.loads(path.read_text())
    metric_path = path.parent / 'metrics.jsonl'
    metrics = [json.loads(line) for line in metric_path.read_text().splitlines()] if metric_path.exists() else []
    result_path = path.parent / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    latest = metrics[-1] if metrics else None
    print(json.dumps({'arm': path.parent.name, 'pid': state['pid'], 'status': state['status'], 'heartbeat_age_seconds': round(now-state['heartbeat'], 1), 'epoch': state.get('epoch'), 'batch': state.get('batch'), 'pulses': state.get('pulses'), 'completed_epochs': len(metrics), 'validation_KL': latest['validation']['apparent']['kl_teacher_student'] if latest else None, 'selected_epoch': result['selected']['epoch'] if result else None, 'error': state.get('error')}))
