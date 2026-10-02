"""CPU-only tests of bounded beta search and asynchronous evidence arrivals."""
import copy

import pytest

from experiments.cifar_beta_stage_decision import FALLBACK, INITIAL, decide


def good(accuracy=.60, loss=1.):
    return dict(validated=True, rejected=False,
                validation=dict(accuracy=accuracy, loss=loss))


def completed_1():
    return {"1": {case: good() for case in ("native_bptt", *INITIAL)}}


def requests(result):
    return {(r["case"], r["stage"]) for r in result["requests"]}


def fallback_1():
    records = completed_1()
    for case in INITIAL:
        records["1"][case] = good(.50)
    return records


def test_initial_requests_and_missing_native_wait():
    assert requests(decide({})) == {(c, "1") for c in ("native_bptt", *INITIAL)}
    result = decide({"1": {c: good() for c in INITIAL}})
    assert result["state"] == "pending"
    assert requests(result) == {("native_bptt", "1")}


def test_partial_initial_round_cannot_promote_and_input_is_unchanged():
    records = {"1": {"native_bptt": good(), INITIAL[0]: good()}}
    snapshot = copy.deepcopy(records)
    assert requests(decide(records)) == {(INITIAL[1], "1")}
    assert records == snapshot


def test_both_initial_passers_advance_and_partial_five_waits():
    records = completed_1()
    assert requests(decide(records)) == {(c, "5") for c in ("native_bptt", *INITIAL)}
    records["5"] = {"native_bptt": good(), INITIAL[0]: good()}
    assert requests(decide(records)) == {(INITIAL[1], "5")}
    records["5"][INITIAL[1]] = good()
    assert requests(decide(records)) == {(c, "10") for c in ("native_bptt", *INITIAL)}


@pytest.mark.parametrize("candidate,passes", [
    (good(.58, 1.05), True), (good(.579, 1.), False),
    (good(.60, 1.051), False), (good(float("nan")), False),
    (good(loss=float("inf")), False), (good(loss=-1), False),
    (dict(validated=True, rejected=True), False),
    (dict(validated=True), False),
])
def test_gate_boundaries_and_candidate_rejection(candidate, passes):
    records = completed_1()
    records["1"][INITIAL[0]] = candidate
    result = decide(records)
    assert result["selected"] == (list(INITIAL) if passes else [INITIAL[1]])


@pytest.mark.parametrize("invalid", [
    good(float("nan")), dict(validated=True, rejected=True),
    dict(validated=True), dict(validated=False), None,
])
def test_invalid_native_blocks(invalid):
    records = completed_1()
    records["1"]["native_bptt"] = invalid
    result = decide(records)
    assert result["state"] == "blocked" and not result["requests"]


def test_unvalidated_candidate_blocks_instead_of_promoting():
    records = completed_1()
    records["1"][INITIAL[0]]["validated"] = False
    assert decide(records)["state"] == "blocked"


def test_five_failure_activates_fallback_and_reuses_native_five():
    records = completed_1()
    records["5"] = {"native_bptt": good(), **{c: good(.50) for c in INITIAL}}
    assert requests(decide(records)) == {(c, "1") for c in FALLBACK}
    records["1"].update({c: good() for c in FALLBACK})
    assert requests(decide(records)) == {(c, "5") for c in FALLBACK[:2]}


def test_wait_for_all_six_fallback_results_before_ranking():
    records = fallback_1()
    records["1"].update({c: good() for c in FALLBACK[:-1]})
    assert requests(decide(records)) == {(FALLBACK[-1], "1")}
    records["1"][FALLBACK[-1]] = good(.61, .9)
    result = decide(records)
    assert result["selected"] == [FALLBACK[-1], FALLBACK[0]]


def test_fallback_ranking_loss_then_accuracy_then_multiplier():
    records = fallback_1()
    records["1"].update({c: good() for c in reversed(FALLBACK)})
    records["1"][FALLBACK[2]] = good(.61)
    records["1"][FALLBACK[3]] = good(.59, .9)
    assert decide(records)["selected"] == [FALLBACK[3], FALLBACK[2]]
    records["1"][FALLBACK[2]] = good()
    assert decide(records)["selected"] == [FALLBACK[3], FALLBACK[0]]


@pytest.mark.parametrize("failure_stage", ["1", "5"])
def test_fallback_exhaustion_never_restarts_search(failure_stage):
    records = fallback_1()
    records["1"].update({c: good(.50 if failure_stage == "1" else .60)
                            for c in FALLBACK})
    if failure_stage == "5":
        records["5"] = {"native_bptt": good(), **{c: good(.50) for c in FALLBACK[:2]}}
    result = decide(records)
    assert result["state"] == "search_exhausted" and not result["requests"]


@pytest.mark.parametrize("branch", ["initial", "fallback"])
def test_ten_epoch_terminal_tree_reuses_native_and_never_searches_again(branch):
    records = completed_1() if branch == "initial" else fallback_1()
    selected = INITIAL if branch == "initial" else FALLBACK[:2]
    if branch == "fallback":
        records["1"].update({c: good() for c in FALLBACK})
    records["5"] = {"native_bptt": good(), selected[0]: good(), selected[1]: good(.50)}
    records["10"] = {"native_bptt": good()}
    assert requests(decide(records)) == {(selected[0], "10")}
    records["10"][selected[0]] = dict(validated=True, rejected=True)
    result = decide(records)
    assert result["state"] == "complete" and not result["requests"]
    assert result["phase"] == f"{branch}_10"


@pytest.mark.parametrize("branch", ["initial", "fallback"])
def test_extended_milestones_wait_for_full_round_and_complete_at_fifty(branch):
    records = completed_1() if branch == "initial" else fallback_1()
    selected = INITIAL if branch == "initial" else FALLBACK[:2]
    if branch == "fallback":
        records["1"].update({c: good() for c in FALLBACK})
    records["5"] = {c: good() for c in ("native_bptt", *selected)}
    for stage in ("10", "20", "30", "50"):
        result = decide(records, final_stage="50")
        assert requests(result) == {(c, stage) for c in ("native_bptt", *selected)}
        assert result["phase"] == f"{branch}_{stage}"
        records[stage] = {"native_bptt": good(), selected[0]: good()}
        assert requests(decide(records, final_stage="50")) == {(selected[1], stage)}
        records[stage][selected[1]] = good()
    records["50"].update({c: good(.40) for c in selected})
    result = decide(records, final_stage="50")
    assert result["state"] == "complete" and not result["requests"]
    assert result["phase"] == f"{branch}_50"


@pytest.mark.parametrize("failure_stage", ["10", "20", "30"])
@pytest.mark.parametrize("branch", ["initial", "fallback"])
def test_late_quality_stop_preserves_evidence_and_never_restarts_search(failure_stage, branch):
    records = completed_1() if branch == "initial" else fallback_1()
    selected = INITIAL if branch == "initial" else FALLBACK[:2]
    if branch == "fallback":
        records["1"].update({c: good() for c in FALLBACK})
    for stage in ("5", "10", "20", "30"):
        records[stage] = {c: good() for c in ("native_bptt", *selected)}
        if stage == failure_stage:
            records[stage].update({c: good(.40) for c in selected})
            break
    snapshot = copy.deepcopy(records)
    result = decide(records, final_stage="50")
    assert result["state"] == "quality_stop" and not result["requests"]
    assert result["phase"] == f"{branch}_{failure_stage}"
    assert result["selected"] == list(selected) and records == snapshot


def test_only_surviving_candidates_extend_past_ten():
    records = {stage: {c: good() for c in ("native_bptt", *INITIAL)}
               for stage in ("1", "5", "10")}
    records["10"][INITIAL[1]] = dict(validated=True, rejected=True)
    assert requests(decide(records, final_stage="50")) == {
        ("native_bptt", "20"), (INITIAL[0], "20")}


@pytest.mark.parametrize("candidate_env", [
    {"torch": "2.11", "cuda": "12.8"},
    {"torch": "2.5", "cuda": "12.8"},
    {"torch": "2.11", "cuda": "12.4"},
    {"torch": "2.11"}, "invalid",
])
def test_environment_comparison_when_both_records_have_environment(candidate_env):
    records = completed_1()
    native_env = {"torch": "2.11", "cuda": "12.8"}
    records["1"]["native_bptt"]["environment"] = native_env
    records["1"][INITIAL[0]]["environment"] = candidate_env
    result = decide(records, final_stage="50")
    assert result["state"] == ("pending" if candidate_env == native_env else "blocked")


def test_environment_mismatch_blocks_later_milestones_too():
    records = {stage: {c: good() for c in ("native_bptt", *INITIAL)}
               for stage in ("1", "5", "10", "20")}
    records["20"]["native_bptt"]["environment"] = {"torch": "2.11", "cuda": "12.8"}
    records["20"][INITIAL[1]]["environment"] = {"torch": "2.5", "cuda": "12.4"}
    assert decide(records, final_stage="50")["state"] == "blocked"


def test_unsupported_final_stage_rejected():
    with pytest.raises(ValueError, match="final_stage"):
        decide({}, final_stage="100")


def test_direct_fallback_requests_matched_native_and_six_cases_only():
    result = decide({}, final_stage="50", start_branch="fallback")
    assert result["phase"] == "fallback_1"
    assert requests(result) == {(case, "1") for case in ("native_bptt", *FALLBACK)}
    assert not any(case in INITIAL for case, stage in requests(result))


def test_direct_fallback_promotes_best_two_without_initial_receipts():
    records = {"1": {case: good() for case in ("native_bptt", *FALLBACK)}}
    records["1"][FALLBACK[2]] = good(.60, .8)
    records["1"][FALLBACK[4]] = good(.61, .9)
    snapshot = copy.deepcopy(records)
    result = decide(records, final_stage="50", start_branch="fallback")
    assert result["selected"] == [FALLBACK[2], FALLBACK[4]]
    assert requests(result) == {(case, "5") for case in ("native_bptt", FALLBACK[2], FALLBACK[4])}
    assert records == snapshot and all(case not in records["1"] for case in INITIAL)


def test_direct_fallback_keeps_fifty_epoch_quality_gates_and_terminal_rule():
    records = {"1": {case: good() for case in ("native_bptt", *FALLBACK)}}
    selected = FALLBACK[:2]
    for stage in ("5", "10", "20", "30", "50"):
        result = decide(records, final_stage="50", start_branch="fallback")
        assert requests(result) == {(case, stage) for case in ("native_bptt", *selected)}
        records[stage] = {case: good() for case in ("native_bptt", *selected)}
        if stage in ("10", "20", "30"):
            failing = copy.deepcopy(records)
            failing[stage].update({case: good(.40) for case in selected})
            stopped = decide(failing, final_stage="50", start_branch="fallback")
            assert stopped["state"] == "quality_stop" and not stopped["requests"]
    records["50"].update({case: good(.40) for case in selected})
    result = decide(records, final_stage="50", start_branch="fallback")
    assert result["state"] == "complete" and result["phase"] == "fallback_50"
    assert not result["requests"]


@pytest.mark.parametrize("stage", ["1", "5"])
def test_direct_fallback_exhausts_without_requiring_initial_receipts(stage):
    records = {"1": {"native_bptt": good(), **{case: good(.40 if stage == "1" else .60) for case in FALLBACK}}}
    if stage == "5":
        records["5"] = {"native_bptt": good(), **{case: good(.40) for case in FALLBACK[:2]}}
    result = decide(records, final_stage="50", start_branch="fallback")
    assert result["state"] == "search_exhausted" and result["phase"] == f"fallback_{stage}"
    assert not result["requests"]


def test_unsupported_start_branch_rejected():
    with pytest.raises(ValueError, match="start_branch"):
        decide({}, start_branch="resume")
