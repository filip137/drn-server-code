"""Prepare the six reduced-current initialization contexts; never launch compute."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'conv123-reduced-current-layerwise-20260928-v1'
CONFIGS = Path('configs/conv/reduced_current_layerwise_20260928')
LEVELS = [1e-6, 1e-5, 1e-4, 3e-4]


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def write(path, value):
    path = ROOT/path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    for depth, target, seconds in ((1, 1, 1200), (2, 4, 2400), (3, 6, 5400)):
        original = json.loads((ROOT/f'configs/conv/relative_noise_advantage_conv{depth}_20260925.json').read_text())
        reference = next(c for c in original['cases'] if c['scheme']=='baseline' and c['relative_target']==target)
        match_path = Path(f'results/conv123-relative-noise-advantage-map-20260925-v1/stage_a/conv{depth}/cases/conv{depth}_baseline_relative{target}/matching.json')
        matching = json.loads((ROOT/match_path).read_text())
        config = copy.deepcopy(original)
        config.update(study_id=STUDY, stage='layerwise', output_root=f'results/{STUDY}/production/conv{depth}',
                      target='akib', budget_seconds=seconds, prior_gpu_seconds=0.,
                      relative_targets=[float(target)], noise_levels=LEVELS, source_commit=commit,
                      tests_and_canaries_skipped_by_user=False, cases=[])
        for scheme, current in (('v1_c0p5', .5), ('v1_c0p25', .25)):
            source = json.loads((ROOT/reference['source_config']).read_text())
            source['model_base']['voltage_amp'] = 1.
            source['model_base']['current_amp'] = current
            source['arm_id'] = f'conv{depth}_{scheme}_initialization'
            source['reduced_current_replay'] = dict(parent_config=reference['source_config'],
                parent_sha256=reference['source_config_sha256'], optimizer_steps_applied=False,
                amplification_only_change=True)
            source_path = CONFIGS/'sources'/f'conv{depth}_{scheme}.json'
            write(source_path, source)
            case = copy.deepcopy(reference)
            case.update(scheme=scheme, name=f'conv{depth}_{scheme}_relative{target}',
                        source_config=str(source_path), source_config_sha256=sha(source_path),
                        injected_beta=matching['injected_beta'], beta_reference_target=float(target),
                        noise_levels=LEVELS,
                        beta_proposal_reference=dict(path=str(match_path), sha256=sha(match_path),
                            scheme='baseline', relative_target=target, inherited_as_proposal_only=True))
            for field in ('calibration','calibration_proof'):
                case.pop(field, None)
            config['cases'].append(case)
        config['expected_layer_comparisons'] = 2*(depth+1)*(36+12*len(LEVELS)*3)
        write(CONFIGS/f'conv{depth}.json', config)
    print(CONFIGS)


if __name__ == '__main__':
    main()
