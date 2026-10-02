import pytest

from experiments.collect_cifar_linear_outputs import check_metrics


def rows():
    return [dict(epoch=i, steps=i * 1407, train_examples=45000, validation_examples=5000,
                 evaluation_split='validation', train_loss=1.0, validation_loss=1.1,
                 train_accuracy=.5, validation_accuracy=.4) for i in range(1, 11)]


def test_complete_coverage_and_accuracy_loss_earliest_tie():
    values = rows()
    values[4]['validation_accuracy'] = values[5]['validation_accuracy'] = .6
    assert check_metrics(values, {'epochs': 10, 'batch_size': 32})['epoch'] == 5
    values[5]['validation_loss'] = 1.0
    assert check_metrics(values, {'epochs': 10, 'batch_size': 32})['epoch'] == 6


@pytest.mark.parametrize('field,value,message', [
    ('steps', 1, 'optimizer steps'), ('validation_examples', 16, 'data coverage'),
    ('evaluation_split', 'official_test', 'evaluation split'),
    ('validation_loss', float('nan'), 'Nonfinite metrics'),
])
def test_incomplete_or_invalid_measurements_fail(field, value, message):
    values = rows()
    values[-1][field] = value
    with pytest.raises(ValueError, match=message):
        check_metrics(values, {'epochs': 10, 'batch_size': 32})


def test_missing_epoch_fails():
    with pytest.raises(ValueError, match='epoch coverage'):
        check_metrics(rows()[:-1], {'epochs': 10, 'batch_size': 32})
