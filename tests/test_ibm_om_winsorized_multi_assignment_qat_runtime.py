from __future__ import annotations

import pytest

from experiments.mnist_relu_drn.ibm_om_winsorized_multi_assignment_qat_runtime import (
    assignment_seed_for_minibatch,
    development_candidate_key,
    pre_final_assignment_seeds,
)


def test_global_minibatch_cycle_alternates_exactly_and_balances_counts() -> None:
    cycle = (86001, 87001)
    observed = [assignment_seed_for_minibatch(cycle, index) for index in range(11)]
    assert observed == [86001, 87001] * 5 + [86001]
    assert observed.count(86001) == 6
    assert observed.count(87001) == 5

    # The ordinal is global, so an odd-sized epoch/chunk does not restart at 86001.
    continued = [
        assignment_seed_for_minibatch(cycle, index) for index in range(11, 15)
    ]
    assert continued == [87001, 86001, 87001, 86001]


@pytest.mark.parametrize("ordinal", [-1, True, 1.5])
def test_global_minibatch_cycle_rejects_invalid_ordinals(ordinal: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        assignment_seed_for_minibatch((86001, 87001), ordinal)  # type: ignore[arg-type]


@pytest.mark.parametrize("cycle", [(), (86001, 86001)])
def test_global_minibatch_cycle_rejects_invalid_assignments(cycle: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="nonempty unique"):
        assignment_seed_for_minibatch(cycle, 0)


def test_final_87003_is_excluded_from_training_and_development_surface() -> None:
    selected = pre_final_assignment_seeds((86001, 87001), 87002, 87003)
    assert selected == (86001, 87001, 87002)
    assert 87003 not in selected

    with pytest.raises(ValueError, match="one final role"):
        pre_final_assignment_seeds((86001, 87001), 87002, 87002)


def test_development_selection_is_correct_then_kl_then_earlier_epoch() -> None:
    baseline = {"student_correct": 4800, "kl_teacher_student": 0.2}
    more_correct = {"student_correct": 4801, "kl_teacher_student": 2.0}
    lower_kl = {"student_correct": 4800, "kl_teacher_student": 0.1}

    assert development_candidate_key(more_correct, 9) > development_candidate_key(
        baseline, 1
    )
    assert development_candidate_key(lower_kl, 9) > development_candidate_key(
        baseline, 1
    )
    assert development_candidate_key(baseline, 1) > development_candidate_key(
        baseline, 2
    )
