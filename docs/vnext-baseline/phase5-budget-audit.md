# Phase 5 — coupled budget audit

Audit only. **No limit was changed**, per the instruction to prove the
query and selection fixes offline before touching public limits.

## What configuration can and cannot reach

`limits_from_settings` builds a `DemoLimits` and overrides exactly five
fields. Every other field uses the dataclass default, and `_MAX` *is*
that same default-constructed dataclass — so the defaults are
simultaneously the values in force and the ceiling configuration may
not exceed.

| Overridable by environment | Hard-coded (environment has no effect) |
| --- | --- |
| `max_runtime_seconds` | `max_rounds` = **1** |
| `runs_per_ip_per_hour` | `max_sources` = 6 |
| `max_concurrent_runs` | `max_sources_per_round` = 6 |
| `max_provider_requests_per_day` | `max_search_queries` = 6 |
| `global_runs_per_day` (derived) | `max_llm_calls` = 20 |
| | `max_provider_requests` = 30 |
| | `max_cloud_calls` = 20 |
| | `max_cloud_input_tokens` = 240,000 |
| | `max_cloud_output_tokens` = 20,000 |
| | `max_cloud_cost_usd` = 0.05 |
| | `max_search_credits` = 8.0 |
| | `max_query_chars` = 300 / `min_query_chars` = 10 |

**Finding 1 — `MAX_RESEARCH_ROUNDS` cannot raise the demo above one
round.** It is not clamped; it is simply never read for the demo path.
`apply_demo_limits` narrows run settings to `state.limits`, whose
`max_rounds` is the hard-coded 1. Anyone setting that variable in the
environment would see no change and no error.

**Finding 2 — raising `max_rounds` alone would still not produce a
second round.** `route_after_coverage` exits to synthesis on *any* of:

```python
round_number >= budget.max_research_rounds
len(completed_queries) >= budget.max_search_queries   # 6
len(sources) >= budget.max_sources                    # 6
```

Round one issues 6 queries and selects 6 sources — both caps are
exhausted by the first pass. So even with `max_rounds = 2`, the second
and third conditions fire and the run synthesises anyway. A second
round is structurally unreachable without reallocating queries and
sources, which is why this was never a one-line change.

## Measured worst case for one round

From the preserved live capture (`examples/live-validation/question-shapes/`,
commit b16ad010, question "What are the main causes of hallucination in
large language models?"):

| Quantity | Measured | Ceiling | Headroom |
| --- | --- | --- | --- |
| provider requests | 14 (13 billable, 1 failed) | 30 | 16 |
| LLM calls | 13 | 20 | 7 |
| search queries | 6 | 6 | **0** |
| sources selected | 6 | 6 | **0** |
| wall clock | 144 s | 240 s | 96 s |
| cost | $0.0083 | $0.05 | $0.042 |

A second live run (RAG vs fine-tuning) measured 13 LLM calls, 146.6 s,
$0.0073 — consistent.

## Derivation of a two-round profile, if one is later wanted

Provider requests and cost have ample headroom; **queries, sources and
runtime are the binding constraints**. The brief's shape — a bounded
first pass reserving capacity for a targeted second — costs nothing
extra in provider budget if the totals are held:

| | round 1 | round 2 (gap-targeted) | total | today |
| --- | --- | --- | --- | --- |
| search queries | 4 | 2 | 6 | 6 in round 1 |
| sources | 4 | 2 | 6 | 6 in round 1 |
| LLM calls | ~10 | ~5 | ~15 | 13 |

Same provider-request and cost envelope, same daily allowance, no
change to `DEMO_PROVIDER_REQUESTS_PER_DAY`.

**The unresolved constraint is runtime.** One round measured 144 s of a
240 s ceiling. Two rounds add a second search/fetch/extract cycle and a
second barrier, and the second round cannot begin until the first
completes. The 96 s of headroom is probably not enough, and the brief
is explicit that 240 s must not become 400 s without measured
cold-verifier and end-to-end evidence. **No such measurement exists**,
so no profile change is proposed here.

## Recommendation

Keep one round. The Phase 4 changes make the existing six sources
better spent rather than needing more of them:

- selection now allocates across sub-questions, so six sources cover up
  to six dimensions instead of six candidates of one dimension — which
  is what produced 1/5 coverage;
- queries are written in the question's own vocabulary, so the six
  pages retrieved should be on-topic explanatory sources rather than
  tangential papers.

Both are the brief's own fallback: "keep one round and make the planner
use fewer, higher-quality, answer-slot-focused subquestions." Whether
they suffice is an empirical question, and the honest way to settle it
is one capped live run on the preserved baseline question, compared
against the preserved artifact — not a limit increase.

## Inconsistency worth recording

`DemoLimits.max_provider_requests_per_day` defaults to 50 while the
deployed dashboard value is 720, and `global_runs_per_day` is derived
as `720 // 30 = 24`. The default is not a ceiling here (this field is
deliberately unclamped, because it states an external fact about the
provider account), so the derived allowance is correct in production.
The dataclass default is simply stale as documentation.
