"""Pure promotion policy for the bounded fresh CIFAR beta experiment.

Records are keyed by stage (``1``, ``5``, ``10``, ``20``, ``30``, ``50``), then case. Only collected
records with ``validated: True`` are evidence; numerical rejections additionally
set ``rejected: True``. Other records carry validation accuracy (fraction) and
loss. Requests deliberately include missing evidence for jobs already running;
the transport owns idempotent submission, collection, and operational recovery.
"""
from collections.abc import Mapping
from math import isfinite


INITIAL = ("beta_0p3", "beta_1")
MULTIPLIERS = {
    "beta_0p0001": .0001, "beta_0p001": .001, "beta_0p01": .01,
    "beta_0p03": .03, "beta_0p1": .1, "beta_3": 3.,
}
FALLBACK = tuple(MULTIPLIERS)


def _metrics(record):
    if not isinstance(record, Mapping) or record.get("rejected") is True:
        return None
    try:
        accuracy = record["validation"]["accuracy"]
        loss = record["validation"]["loss"]
        if isinstance(accuracy, bool) or isinstance(loss, bool):
            return None
        if isfinite(accuracy) and isfinite(loss) and 0 <= accuracy <= 1 and loss >= 0:
            return accuracy, loss
    except (KeyError, TypeError, ValueError, OverflowError):
        pass
    return None


def decide(records, final_stage="10", start_branch="initial"):
    """Return missing requests, state, phase and selected case IDs; never mutate.

    A rejected or unusable native control blocks comparisons. Present records
    lacking collection validation also block rather than silently authorizing
    progression. Candidate numerical rejection/nonfinite metrics fail the gate.
    With final_stage="50", passing cases continue through epochs 20, 30 and 50;
    failure from epoch 10 onward stops for quality without another search.
    Completion at the final milestone is terminal regardless of its comparison.
    start_branch="fallback" starts with the six fallback cases and their matched
    native control, without requiring initial-case receipts from another cohort.
    """
    if final_stage not in ("10", "50"):
        raise ValueError("final_stage must be '10' or '50'")
    if start_branch not in ("initial", "fallback"):
        raise ValueError("start_branch must be 'initial' or 'fallback'")

    def outcome(state, phase, selected, requests=()):
        return dict(state=state, phase=phase, selected=list(selected),
                    requests=list(requests))

    def round_result(stage, cases, phase):
        stage_records = records.get(stage, {})
        expected = ("native_bptt", *cases)
        present = [stage_records[c] for c in expected if c in stage_records]
        if any(not isinstance(r, Mapping) or r.get("validated") is not True
               for r in present):
            return outcome("blocked", phase, cases), []
        native = _metrics(stage_records.get("native_bptt"))
        if "native_bptt" in stage_records and native is None:
            return outcome("blocked", phase, cases), []
        native_env = stage_records.get("native_bptt", {}).get("environment")
        for record in present:
            env = record.get("environment")
            if native_env is not None and env is not None:
                if (not isinstance(native_env, Mapping) or not isinstance(env, Mapping)
                        or any(k not in env or k not in native_env or env[k] != native_env[k]
                               for k in ("torch", "cuda"))):
                    return outcome("blocked", phase, cases), []
        requests = [dict(case=c, stage=stage) for c in expected if c not in stage_records]
        if requests:
            return outcome("pending", phase, cases, requests), []
        passing = []
        for case in cases:
            metrics = _metrics(stage_records[case])
            if metrics and metrics[0] >= native[0] - .02 and metrics[1] <= native[1] * 1.05:
                passing.append(case)
        return None, passing

    branches = (("initial", INITIAL), ("fallback", FALLBACK))
    if start_branch == "fallback":
        branches = branches[1:]
    for branch, candidates in branches:
        phase = f"{branch}_1"
        pending, selected = round_result("1", candidates, phase)
        if pending:
            return pending
        if branch == "fallback":
            selected.sort(key=lambda c: (_metrics(records["1"][c])[1],
                                         -_metrics(records["1"][c])[0], MULTIPLIERS[c]))
            selected = selected[:2]
        if not selected:
            if branch == "initial":
                continue
            return outcome("search_exhausted", phase, [])
        phase = f"{branch}_5"
        pending, promoted = round_result("5", selected, phase)
        if pending:
            return pending
        if not promoted:
            if branch == "initial":
                continue
            return outcome("search_exhausted", phase, [])
        for stage in (("10",) if final_stage == "10" else ("10", "20", "30", "50")):
            phase = f"{branch}_{stage}"
            pending, passing = round_result(stage, promoted, phase)
            if pending:
                return pending
            if stage == final_stage:
                return outcome("complete", phase, promoted)
            if not passing:
                return outcome("quality_stop", phase, promoted)
            promoted = passing
