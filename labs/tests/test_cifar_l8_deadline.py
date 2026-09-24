from datetime import timezone

import pytest

from experiments.train_cifar_l8_analog import epoch_admission_seconds, parse_stop_before


def test_epoch_admission_accounts_for_slow_recent_epochs_and_shutdown():
    assert epoch_admission_seconds([], 900, 120) == 1245
    assert epoch_admission_seconds([750, 1100, 800], 900, 120) == 1495
    assert epoch_admission_seconds([1100, 750, 800, 760], 900, 120) == 1120


def test_deadline_timezone_is_explicit_and_converted_to_utc():
    parsed = parse_stop_before('2026-09-24T08:00:00+02:00')
    assert parsed.hour == 6 and parsed.tzinfo == timezone.utc
    assert parsed == parse_stop_before('2026-09-24T06:00:00Z')
    with pytest.raises(Exception, match='explicit timezone'):
        parse_stop_before('2026-09-24T08:00:00')
