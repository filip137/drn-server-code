# Experiment Catalog and Minimal Launch Request

The catalog is the front door for experiment requests. It maps a stable
experiment identity or a short description to the current protocol, config,
plan, and launcher without copying their scientific values.

For a user-facing form that does not expose hashes, receipts, or environment
plumbing, start from
[`experiment_request_template.md`](experiment_request_template.md). Completed
intake forms and agent-resolved summaries live under
[`experiment_requests/`](experiment_requests/README.md).

The catalog is not launch authority. A request can launch work only when it
resolves uniquely, receives the applicable scientific and execution review,
has no conflicting live run, and passes the required preflight gates. An
ordinary diagnostic continuation may use the generated scientific-only study;
it does not need a duplicate legacy mixed plan.

## Mandatory intake before launch preparation

If the user says “launch,” “run,” “schedule,” or “continue” without a
completed request, the agent first uses read-only context to prefill
[`experiment_request_template.md`](experiment_request_template.md), presents
its compact conversational form, and asks the user to complete or correct it.
The agent must not create a plan or tracker entry, run preflight, stage files,
or launch while waiting.

The user may answer `inherit from parent`. Known parent values must be
prefilled rather than requested again. Even when every field is known, the
agent presents the resolved form for one confirmation. Intake answers alone
are not approval; when the same presentation includes the complete resolved
scientific summary and execution proposal, one explicit `approved` response
may approve both without another round trip.

## Inspect the catalog

```bash
python -m experiments.experiment_catalog list
python -m experiments.experiment_catalog show EXPERIMENT_ID
python -m experiments.experiment_catalog resolve QUERY
python -m experiments.experiment_catalog check
python -m experiments.experiment_flow REQUEST.md --repo-root .
```

The text resolver returns `resolved`, `ambiguous`, or `not_found` and never
silently selects a nearest match. The structured flow additionally returns a
derived parent subset when the immutable parent and differences resolve
unambiguously. `resolved` or `derived` means routing succeeded; neither means
the run is already clear to launch.

## Minimal request

For an existing catalog entry with an approved plan, this is sufficient to
ask the agent to prefill the intake:

```text
Launch: <catalog experiment-id>
Plan: unchanged
Compute: auto
```

`Plan: unchanged` explicitly adopts the linked scientific definition byte for
byte. The agent still shows the prefilled intake and resolved summary for
confirmation. The simplified continuation flow then generates target plans
and one bounded validation dispatch request per target. It creates and
maintains one compact `docs/current_experiments.md` entry automatically; the
user is not asked to write tracker syntax, hashes, receipts, paths, or
environment fields.

For a new study or any material change, use the expanded form:

```text
Launch: <catalog experiment-id or "new">
Plan: successor; changes=<what changes and what remains fixed>
Hypothesis: <control> vs <treatments>; expect <direction>.
Sweep: <axis>=[ordered values]; cases=[...]; seeds=[...]; expand=<none|rule>; cap=<total jobs>.
Decision: metric=<...>; split=<...>; checkpoint=<...>; seeds=<aggregation>; compare=<...>; meaningful=<threshold + units>.
Outcomes: support=<...>; reject=<...>; inconclusive=<...>; complete=<required coverage>; exclude=<...>; failure=<...>.
Evidence: diagnostic | paper-facing | historical replay
Compute: auto | <preference>
```

The structured template is preferred for derived studies, continuations,
multi-host execution, or any request that is difficult to describe in this
single copy/paste block. For a fixed ordinary-diagnostic continuation, the
generated `experiment-resolved-study/v1` object is the plan and
`experiments.functional_dispatch` is the sole supervisor. A real one-batch
smoke runs on every target; exact environment-version matching and a repeated
`T/K` gate are not required when the accepted scientific operating point is
unchanged.

The agent may replace `Sweep`, `Decision`, or `Outcomes` with `plan` only when
a linked immutable approved plan already binds that complete section. A
general protocol is not sufficient. An ambiguous query requires
disambiguation. An absent query is a new workflow, not permission to use the
nearest suggestion. In either case, the agent should preserve all supplied
choices, ask only for missing decisions, and freeze a new plan before launch.
