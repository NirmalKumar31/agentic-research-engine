# Release audit

Candidate-level validation of the publication gate, at the unit the
system actually publishes: the **atomic generated claim**.

This is separate from
[`examples/verifier-calibration/`](../verifier-calibration/), and the
distinction matters. Those thirty cases are **development calibration
data** — they shaped three generative verifier designs and the NLI
replacement, so agreement measured on them is partly fitted. They are
kept, unmodified, as the historical record of how the verifier was
built.

This directory is **release validation**: the claims three real runs
actually produced, under the final synthesis prompt and the final gate,
reviewed against their own evidence.

## The release metric

> **Unsupported published atomic claims must equal zero.**

That is a human judgement and nothing here computes it. `build_audit.py`
produces the reviewable list — every candidate with its complete claim
text, its cited quotes, every pairwise NLI score, every guard result and
whether it reached the published report — with a `human_review` field
left null for a reviewer to fill.

Alongside it: raw candidates, deduplicated, checked, published,
withheld, evidence-only excerpts, and the publication rate.

## What a record contains

Every unique post-deduplication substantive candidate, whether or not it
was checked:

- complete claim text, never truncated, and its kind
- each cited evidence id with its quote, source, page and match type
- per-evidence entailment / neutral / contradiction probabilities
- per-evidence guard results, with the names of any that failed
- best guard-passing evidence id and its entailment
- model id, pinned revision and support threshold
- `publishable`, the diagnostic verdict, and the withhold reason
- whether the claim appears in the published report
- any compound-claim markers, so a fused claim that publishes is visible

`gate_disagreements` lists any claim present in the report that the gate
marked unpublishable. It should always be empty; it exists so that if
the gate and the renderer ever disagree, the audit says so rather than
hiding it.

## Regenerating

```bash
python examples/release-audit/build_audit.py
```

Reads the committed recordings, so it describes the same runs the demo
site replays. Changing the numbers means re-recording, not re-running
this.
