import json
from pathlib import Path

import pytest
import torch

from experiments import reporting
from experiments.analyze_cifar_digital import analyze
from experiments.train_cifar_digital_l12 import DigitalL12, make_optimizer, save_checkpoint


def prepare_study(root, *, invalid_steps=False, nonfinite_optimizer=False):
    config = json.loads(Path('configs/cifar/digital_l8_linear_outputs_20261002.json').read_text())
    config.update(blocks=[[2, 2, 2], [3, 3, 3], [4, 4]], input_shape=[3, 8, 8],
                  readout_features=4, epochs=2)
    model = DigitalL12(config)
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    smoke = root / 'smoke'
    manifest = dict(study_id=config['study_id'], run_id='smoke', arm_id=config['arm_id'],
                    resolved_config=config, dataset={'official_test_read': False}, smoke=True,
                    evidence_class='unit_test_fixture')
    reporting.start_run(smoke, manifest)
    reporting.complete_run(smoke, terminal_metrics={})
    run = root / 'production-seed0'
    reporting.start_run(run, dict(manifest, run_id=run.name, smoke=False))
    reporting.atomic_write_json(run / 'artifacts/split.json',
                                {'train': list(range(45000)), 'validation': list(range(45000, 50000))})
    for name, epoch in [('initial', 0), ('best', 1), ('final', 2)]:
        if nonfinite_optimizer and name == 'final':
            optimizer.state[next(model.parameters())]['exp_avg'] = torch.tensor(float('nan'))
        save_checkpoint(run / f'checkpoints/{name}_model.pt', model, optimizer, scheduler, epoch, config)
    for epoch in (1, 2):
        reporting.append_metric(run / 'metrics.jsonl', {
            'epoch': epoch, 'evaluation_split': 'validation', 'train_examples': 45000,
            'validation_examples': 5000, 'optimizer_steps': 88 * epoch - int(invalid_steps),
            'train_accuracy': .5, 'train_loss': 1.5, 'validation_accuracy': .4,
            'validation_loss': 1.6, 'peak_memory_reserved_bytes': 1000000})
    reporting.complete_run(run, terminal_metrics={'best_epoch': 1, 'best_validation_accuracy': .4,
                                                 'final_validation_accuracy': .4, 'optimizer_steps': 176},
                           completion={'criteria_met': True, 'official_test_read': False})


def test_collector_validates_coverage_ties_checkpoints_and_exports_jpg(tmp_path):
    prepare_study(tmp_path)
    summary = analyze(tmp_path)
    assert summary['verification']['checkpoint_epochs'] == {'initial': 0, 'best': 1, 'final': 2}
    assert summary['verification']['steps'] == 176
    assert (tmp_path / 'analysis/learning_curves.jpg').stat().st_size > 1000
    assert summary['official_test_read'] is False


@pytest.mark.parametrize('defect, message', [
    ('invalid_steps', 'Optimizer step coverage mismatch'),
    ('nonfinite_optimizer', 'nonfinite checkpoint'),
])
def test_collector_rejects_semantic_failures_despite_valid_bundle(tmp_path, defect, message):
    prepare_study(tmp_path, **{defect: True})
    with pytest.raises(ValueError, match=message):
        analyze(tmp_path)
    assert not (tmp_path / 'analysis/summary.json').exists()
