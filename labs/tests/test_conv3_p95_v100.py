"""Three focused launch checks for the approved six-case V100 comparison."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch

from experiments.epoch_continuation import EpochContinuation
from experiments.exact_run import selected_configs

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / 'experiments/run_conv3_p95_read_noise_continuation_v100.sh'
CONFIGS = ROOT / 'configs/conv/eqprop_conv3_p95_read_noise_1em3_20260921_v1'


def test_syntax_and_rejection_before_payload(tmp_path):
    subprocess.run(['bash', '-n', str(WRAPPER)], check=True)
    config_dir = tmp_path / 'configs'
    config_dir.mkdir()
    (config_dir / 'config-names.txt').write_text(''.join(f'{i}.json\n' for i in range(6)))
    for chunk, index in [('3', '0'), ('0', '6'), ('0', '-1'), ('0', None)]:
        env = os.environ.copy()
        env.pop('SLURM_ARRAY_TASK_ID', None)
        if index is not None:
            env['SLURM_ARRAY_TASK_ID'] = index
        out = tmp_path / 'out'
        result = subprocess.run(['bash', str(WRAPPER), str(tmp_path / 'source'),
                                 str(config_dir), str(out), str(tmp_path / 'data'), chunk],
                                env=env, capture_output=True, text=True)
        assert result.returncode != 0
        assert not out.exists()


def test_frozen_six_case_scope_and_inheritance():
    names = (CONFIGS / 'config-names.txt').read_text().splitlines()
    assert len(names) == len(set(names)) == 6
    for i, name in enumerate(names):
        scheme = ('baseline', 'ours', 'legacy')[i % 3]
        sigma = .001 if i < 3 else 0.
        cfg = json.loads((CONFIGS / name).read_text())
        parent = json.loads((ROOT / cfg['qualification_source']['parent_refined_config']).read_text())
        assert cfg['seed'] == 0 and cfg['lab']['epochs'] == 30
        assert cfg['eqprop']['endpoint_read_noise_std'] == sigma
        assert cfg['eqprop']['injected_beta_B'] == {
            'baseline': 147.682614594, 'ours': 2.49274796756, 'legacy': 2.81845428732}[scheme]
        assert cfg['beta'] == parent['beta']
        assert cfg['evaluation']['official_test']['policy'] == 'disabled'
        for key in ('model_base', 'model_overrides', 'initialization', 'init_checkpoint_path',
                    'lr', 'learning_rates_by_parameter', 'optimizer', 'parameter_order',
                    'datasets', 'energy_minimizer', 'batch_state_policy', 'runtime_dtype'):
            assert cfg[key] == parent[key], key
        noise = dict(cfg['eqprop']); noise['endpoint_read_noise_std'] = 0.
        assert noise == parent['eqprop']
    subprocess.run(['sha256sum', '--check', '--quiet', 'SHA256SUMS'], cwd=CONFIGS, check=True)


def test_single_runtime_preflight_and_continuation_guards(tmp_path):
    text = WRAPPER.read_text()
    preflight = text.split("if [[ $mode == preflight ]]; then\n  python - <<'PYPREFLIGHT'", 1)[1].split('PYPREFLIGHT\n', 1)[0]
    assert 'import experiments.exact_run' in preflight and 'import labs.mnist_train' in preflight
    assert 'cuda.device' not in preflight and 'open(' not in preflight
    assert text.count('python -m experiments.exact_run') == 1
    assert 'experiments.exact_run "$config" --index 0' in text
    assert 'P95_WRAPPER_SHA256' in text and 'mkdir "$segment/worker_started"' in text
    assert 'Wrong continuation epoch' in text and 'Duplicate first chunk' in text
    assert '7140s' in text and 'chunk_epochs=10' in text
    assert 'canary_task in 0 1 2 3 4 5' in text
    cfg = tmp_path / 'one.json'; cfg.write_text('{}')
    for i in range(6):
        assert selected_configs([cfg], 0, environ={'SLURM_ARRAY_TASK_ID': str(i)}) == [(0, cfg)]

    p = SimpleNamespace(name='weight', state=torch.nn.Parameter(torch.tensor([.4], dtype=torch.float64)))
    opt = torch.optim.Adam([p.state], lr=.01)
    p.state.grad = torch.ones_like(p.state); opt.step()
    loaders = SimpleNamespace(_train_sampler=SimpleNamespace(generator=torch.Generator().manual_seed(4)),
                              _worker_generator=torch.Generator().manual_seed(8))
    estimator = SimpleNamespace(_endpoint_read_noise_generators={'cpu': torch.Generator().manual_seed(2)},
                                endpoint_read_noise_draw_count=16)
    kwargs = dict(run_dir=tmp_path, config={'batch_state_policy': 'reset_each_batch',
                  'evaluation': {'official_test': {'policy': 'disabled'}}}, epochs=30,
                  chunk_epochs=10, parameters=[p], optimizer=opt, estimator=estimator,
                  loaders=loaders, recorder=None, contract={'source': 'frozen'})
    continuation = EpochContinuation(**kwargs)
    saved_weight = p.state.detach().clone()
    saved_rng = loaders._train_sampler.generator.get_state()
    saved_noise = estimator._endpoint_read_noise_generators['cpu'].get_state()
    assert continuation.pause_after(10, history={'loss': [1.]}, best_accuracy=.5, best_epoch=1)
    with torch.no_grad(): p.state.fill_(99)
    opt.state[p.state]['step'].fill_(99)
    loaders._train_sampler.generator.manual_seed(999)
    restored = EpochContinuation(**kwargs)
    assert restored.restore()['epoch'] == 10
    assert torch.equal(p.state, saved_weight) and opt.state[p.state]['step'].item() == 1
    assert torch.equal(loaders._train_sampler.generator.get_state(), saved_rng)
    assert torch.equal(estimator._endpoint_read_noise_generators['cpu'].get_state(), saved_noise)
    assert estimator._endpoint_read_noise_draw_count == 16
