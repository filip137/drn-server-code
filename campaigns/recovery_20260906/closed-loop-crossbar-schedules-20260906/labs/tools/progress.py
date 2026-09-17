"""Compact artifact progress for all declared arms."""
import json
from pathlib import Path
import time

root=Path(__file__).resolve().parents[2]
rows=[]
for schedule in ('constant','exponential'):
    for condition in ('healthy','faulted'):
        for writer in ('open_loop','closed_loop_pv'):
            name=f'{condition}-{schedule}-{writer}'
            path=root/'runs'/name
            row={'arm':name,'status':'queued'}
            if (path/'status.json').exists():
                status=json.loads((path/'status.json').read_text())
                records=[json.loads(s) for s in (path/'metrics.jsonl').read_text().splitlines()] if (path/'metrics.jsonl').exists() else []
                row.update(status=status['status'],epoch=status.get('epoch'),batch=status.get('batch'),
                           pulses=status.get('pulses'),heartbeat_age=round(time.time()-status['heartbeat'],1))
                if records:
                    last=records[-1]
                    if status['status']=='complete':
                        row.update(epoch=last['epoch'],batch=last['batches'],pulses=last['pulses'])
                    row.update(completed_epochs=len(records),validation_kl=last['validation']['apparent']['kl_teacher_student'],
                               last_epoch_seconds=round(last['seconds'],1),pulsed_cells=last['pulsed_cells'],
                               maximum_target_error=last.get('target_error_max'))
                if 'error' in status:row['error']=status['error']
            rows.append(row)
print(json.dumps(rows,indent=2))
with (root/'watchdog_observations.jsonl').open('a') as f:
    f.write(json.dumps({'time':time.time(),'artifacts':rows})+'\n')
