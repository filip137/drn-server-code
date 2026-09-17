"""Read-only target, physical-bound and pulse-cap diagnostics at epoch three."""
from pathlib import Path
import json
import torch

torch.set_num_threads(1)
root = Path(__file__).resolve().parents[2]
plan = json.loads((root / 'plan.json').read_text())
rows = []
for architecture in ('drn', 'crossbar'):
    for condition in ('healthy', 'faulted'):
        path = root / 'runs' / f'{architecture}-{condition}-closed_loop_pv' / 'continuation.pt'
        if not path.exists():
            continue
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        if checkpoint['epoch'] != 3:
            continue
        state = checkpoint['state']
        optimizer, plant = state['optimizer'], state['plant']
        target, count = optimizer['target'], optimizer['pulse_count']
        if architecture == 'drn':
            apparent = (plant['apparent_raw_a'] + 1) / 2
            persistent = (plant['persistent_raw_a'] + 1) / 2
            corrupt = plant['active_corrupt']
            lower, upper = torch.zeros_like(target), torch.ones_like(target)
            lower[corrupt] = persistent[corrupt]
            upper[corrupt] = persistent[corrupt]
        else:
            apparent, persistent = plant['apparent'], plant['persistent']
            corrupt = plant['post_deployment_fault_mask']
            origin = torch.load(plan['crossbar'][f'{condition}_p0'], map_location='cpu', weights_only=False)
            population = origin['healthy_population']
            lower = population['min_bound'] - population['reference']
            upper = population['max_bound'] - population['reference']
            lower[corrupt] = persistent[corrupt]
            upper[corrupt] = persistent[corrupt]
        tolerance = optimizer['tolerance']
        groups = {}
        for name, mask in [('all', torch.ones_like(corrupt)), ('healthy_cells', ~corrupt), ('fault_cells', corrupt)]:
            n = int(mask.sum())
            if not n:
                groups[name] = {'cells': 0}
                continue
            groups[name] = {
                'cells': n, 'pulsed_cells': int(((count > 0) & mask).sum()),
                'pulses': int(count[mask].sum()), 'maximum_pulses': int(count[mask].max()),
                'cells_at_640_cap': int(((count >= 640) & mask).sum()),
                'apparent_within_tolerance_fraction': float(((apparent-target).abs()[mask] <= tolerance).double().mean()),
                'persistent_within_tolerance_fraction': float(((persistent-target).abs()[mask] <= tolerance).double().mean()),
                'desired_target_outside_true_bounds_fraction': float(((target < lower) | (target > upper))[mask].double().mean()),
                'apparent_target_RMS': float((apparent-target)[mask].double().square().mean().sqrt()),
                'persistent_target_RMS': float((persistent-target)[mask].double().square().mean().sqrt())}
        rows.append({'architecture': architecture, 'condition': condition, 'epoch': 3, 'verify_tolerance': tolerance, 'verify_reads': optimizer['verify_reads'], 'cumulative_burst_exhaustions': optimizer['burst_exhaustions'], 'cumulative_cap_block_events': optimizer['cap_blocks'], 'groups': groups})
(root / 'analysis').mkdir(exist_ok=True)
(root / 'analysis' / 'writer_diagnostics.json').write_text(json.dumps(rows, indent=2)+'\n')
print(json.dumps(rows, indent=2))
