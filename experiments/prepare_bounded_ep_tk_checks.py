"""Run the authorized larger-T/K baseline diagnostics on fixed validation cohorts."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.reporting import validate_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'results/paper-training-completion-20260911-v1'
CHECKS = STUDY / 'checks/baseline_tk_20260913'
CEILINGS = ('1em4', '5em4', '1em3')
LADDER = {'conv2': (6, 12, 24, 48), 'conv3': (8, 16, 32, 64)}


def run_case(arch, ceiling, t, k, beta, *, device, smoke=False):
    template = STUDY / f'checks/bounded/configs/{arch}_baseline_gmax_{ceiling}_decade4.json'
    config = json.loads(template.read_text())
    source = CHECKS / 'source'
    config['source_contract']['runtime_source_root'] = str(source)
    for name in config['source_contract']['runtime_code_sha256']:
        config['source_contract']['runtime_code_sha256'][name] = hashlib.sha256((source/name).read_bytes()).hexdigest()
    config['scientific_question'] = 'Does additional equilibrium relaxation resolve bounded-baseline small-beta gradient disagreement?'
    config['evidence_class'] = 'ordinary_mnist_baseline_tk_diagnostic'
    case = config['cases'][0]
    case.update(replay_T=t, replay_K=k, base_beta=beta, injected_beta=beta)
    tag = f'{arch}_{ceiling}_t{t}_k{k}_beta{beta:g}' + ('_smoke' if smoke else '')
    path = CHECKS / 'configs' / (tag+'.json')
    serialized = json.dumps(config, indent=2, sort_keys=True)+'\n'
    if path.exists():
        assert path.read_text() == serialized, 'Diagnostic config changed'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized)
    output = CHECKS / 'runs' / tag
    command = [sys.executable, str(source/'experiments/analyze_conv123_zero_bias_eqprop_bptt_gradient_gate.py'),
               '--config', str(path), '--output-root', str(output.parent), '--run-id', tag,
               '--device', device, '--target', f'main:{device}']
    if smoke:
        command.append('--smoke')
    started = time.monotonic()
    if not (output/'result.json').exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        with (output.parent/(tag+'.log')).open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    assert not validate_run(output), str(output)
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['configuration']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    terminal = json.loads((output/'result.json').read_text())['terminal_metrics']
    assert terminal['official_test_read'] is False and terminal['source_bytes_unchanged']
    record = dict(architecture=arch, ceiling=ceiling, T=t, K=k, injected_beta=beta,
                  run=str(output.relative_to(ROOT)), result_sha256=hashlib.sha256((output/'result.json').read_bytes()).hexdigest(),
                  elapsed_seconds=time.monotonic()-started, command=command, **terminal)
    print(json.dumps({key:record[key] for key in ('architecture','ceiling','T','K','injected_beta','minimum_cosine','maximum_symmetric_norm_delta','unqualified_launch_gate_all_passed')}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        run_case('conv3','1em3',16,16,.01,device=args.device,smoke=True)
        return
    report = {'official_test_read':False, 'training_configs_changed':False,
              'iteration_ladder':LADDER, 'probe_beta':.01, 'larger_beta_ladder':[100.,10.,1.,.1],
              'groups':[], 'axis_diagnostics':[]}
    def save():
        (CHECKS/'selection.json').write_text(json.dumps(report,indent=2)+'\n')
    for arch, ladder in LADDER.items():
        group = dict(architecture=arch, attempts=[], status='running', selected_T=None, selected_K=None, selected_beta=None)
        report['groups'].append(group)
        for tk in ladder:
            rows = [run_case(arch,c,tk,tk,.01,device=args.device) for c in CEILINGS]
            group['attempts'].append(dict(T=tk,K=tk,beta=.01,ceilings=rows))
            save()
            if all(row['unqualified_launch_gate_all_passed'] for row in rows):
                selected = .01
                for beta in (100.,10.,1.,.1):
                    beta_rows = [run_case(arch,c,tk,tk,beta,device=args.device) for c in CEILINGS]
                    group['attempts'].append(dict(T=tk,K=tk,beta=beta,ceilings=beta_rows))
                    save()
                    if all(row['unqualified_launch_gate_all_passed'] for row in beta_rows):
                        selected=beta
                        break
                group.update(status='numerically_qualified_diagnostic_tk', selected_T=tk, selected_K=tk, selected_beta=selected)
                save()
                break
        else:
            group['status']='unresolved_at_tested_tk_and_probe_beta'
            save()
        # Separate the effect of additional free-phase and gradient iterations
        # at the previously failing trained ceiling; never substitute these for full gates.
        ceiling = '1em4' if arch == 'conv2' else '1em3'
        native, doubled = ladder[:2]
        for t,k in ((doubled,native),(native,doubled)):
            report['axis_diagnostics'].append(run_case(arch,ceiling,t,k,.01,device=args.device))
            save()
    print('DIAGNOSTIC_COMPLETE', str(CHECKS/'selection.json'), flush=True)


if __name__ == '__main__':
    main()
