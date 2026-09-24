"""Collect the Akib shard and reconcile it with the local Conv3 measurements."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess

from experiments.reporting import atomic_write_json, sha256_file, utc_now
from experiments.run_beta_rule_comparison import OUT, ROOT, checked, read, verify_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    if args.collect:
        remote = '/home/filiposana/server_code/results/' + OUT.name
        ssh = 'ssh -F /home/filip/.ssh/config -o BatchMode=yes'
        # Architecture placement makes remote/local production names disjoint.
        for source, destination in (
            ('production/', 'production/'), ('logs/', 'akib_logs/'),
            ('akib_execution.json', 'akib_execution.json'), ('launcher/', 'akib_launcher/'),
        ):
            subprocess.run(['rsync', '-a', '-e', ssh, f'akibscomputer:{remote}/{source}',
                            str(OUT / destination)], check=True)
    verify_inputs()
    if sha256_file(OUT / 'plan.frozen.md') != read(OUT / 'inputs.json')['plan_sha256']:
        raise ValueError('Original frozen scientific plan changed.')
    local = read(OUT / 'local_execution.json')
    remote = read(OUT / 'akib_execution.json')
    originals = {c['name']: c for c in read(OUT / 'cases.json')}
    transported = {c['name']: c for c in read(OUT / 'akib_stage/shard.json')['cases']}
    runs = []
    seen = set()
    for shard in (local, remote):
        for case in shard['runs']:
            name = case['name']
            if name in seen or name not in originals:
                raise ValueError('Duplicate or unexpected case.')
            seen.add(name)
            original = originals[name]
            if shard is remote:
                transport = transported[name]
                path = OUT / 'akib_stage/configs' / Path(transport['config']).name
                if sha256_file(path) != case['sha256'] or case['sha256'] != transport['sha256']:
                    raise ValueError('Transport config identity differs.')
                restored = copy.deepcopy(read(path)); frozen = read(ROOT / original['config'])
                for key in ('run_dir', 'initializer_checkpoint_path'):
                    restored['cases'][0][key] = frozen['cases'][0][key]
                for key in ('runtime_source_root', 'source_study_root'):
                    restored['source_contract'][key] = frozen['source_contract'][key]
                restored['dataset']['root'] = frozen['dataset']['root']
                if restored != frozen:
                    raise ValueError('A scientific field changed during transport.')
                mapped = dict(transport, config=str(path.relative_to(ROOT)))
            else:
                if case['sha256'] != original['sha256']:
                    raise ValueError('Local config identity differs.')
                mapped = original
            verified = checked(mapped, OUT / 'production' / name)
            if verified['result_sha256'] != case['result_sha256']:
                raise ValueError('Local copy differs from source result.')
            runs.append(dict(verified, elapsed_seconds=case['elapsed_seconds'], target=shard['target']))
    complete = local['state'] == remote['state'] == 'complete'
    if complete and seen != set(originals):
        raise ValueError('Terminal coverage mismatch.')
    # Reserve/charge a conservative full minute for the <10-second Akib smoke,
    # balanced by its explicit subtraction from the local three-hour cap.
    charge = local['charged_gpu_hours'] + remote['charged_gpu_hours'] + 1/60
    if charge > 6:
        raise ValueError('Combined physical GPU budget exceeded.')
    state = 'complete' if complete else ('failed' if 'failed' in (local['state'], remote['state']) else 'running')
    summary = dict(study_id=OUT.name, state=state, expected_cases=153, runs=runs,
        current_cases=[s['current_case'] for s in (local, remote) if s['current_case']],
        targets=[s['target'] for s in (local, remote)], updated_at=utc_now(),
        charged_gpu_hours=charge, cap_gpu_hours=6, remote_smoke_charged_seconds=60,
        started_at=read(OUT / 'execution.single_worker.json')['started_at'],
        official_test_read=False, training_started=False,
        transport_scientific_equivalence_verified=True)
    if complete:
        summary['completed_at'] = utc_now()
    atomic_write_json(OUT / 'execution.json', summary)
    print(json.dumps(dict(state=state, complete=len(runs), expected=153,
        current=summary['current_cases'], charged_gpu_hours=charge)))


if __name__ == '__main__':
    main()
