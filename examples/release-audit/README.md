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

## Independent reviewer join

`reviewer_packet.py` hands the same 21 candidates to someone other than
the person who built the verifier -- `reviewer-packet.json` carries only
the claim and its single selected quote, with the automated verdict,
score, guard result, publication decision and prior label all
structurally absent (asserted at build time, not just omitted by
convention).

```bash
python examples/release-audit/reviewer_packet.py build   # writes the packet + a blank reviewer-labels.json
#   ... a reviewer fills reviewer-labels.json with
#       supported / unsupported / uncertain, one per case_id ...
python examples/release-audit/reviewer_packet.py join     # writes reviewer-audit.json
```

`join` validates the returned labels (exactly 21, matching the packet's
`case_id`s, every value one of the three allowed labels) before it joins
anything to the hidden publication outcome. The output,
`reviewer-audit.json`, carries a confusion matrix, every published claim
labelled `unsupported` or `uncertain`, a non-identifying process
attestation, and an explicit note that one external reviewer over a
fixed 21-case set is independent release validation, not a statistical
benchmark.

This is a different thing from `blind-audit.json`: that is a blinded
*self*-review (the same person who built the verifier, with the verdict
hidden from them). This is a second person, with no automated outcome
ever visible to them at all.

### One disputed case, a second opinion

`reviewer-audit.json` found one published claim the independent
reviewer labelled unsupported (tracked internally in issue #32).
`second_opinion.py` hands that one case, and nothing else, to a second
adjudicator -- not the first reviewer's label, not the system's score,
guard result or publication decision, not which run or audit it came
from, and not a mention of issue #32 or any model/provider name. The
outward-facing files (`evidence-review-packet.json`,
`evidence-review-label.json`) use a neutral random case id instead of
the repo's own case naming, specifically so neither file hints at the
context this script itself knows about.

A build-time check (`FORBIDDEN_TERMS` in `second_opinion.py`) scans the
packet's own `description` and case data for exactly this class of
leak before writing anything to disk. It exists because an earlier
version of this packet failed at the one thing it was for: its
hand-written description stated outright that "a first independent
reviewer labelled this claim unsupported; the system published it" --
caught before any human received it, not by a test, because the test
didn't exist yet. It does now.

```bash
python examples/release-audit/second_opinion.py build   # writes evidence-review-packet.json + a blank label
#   ... give the packet to a second adjudicator, who fills evidence-review-label.json ...
python examples/release-audit/second_opinion.py join     # writes evidence-review-result.json
```

`join` records both labels side by side and what they imply -- it does
not decide anything itself. If the second adjudicator's label is also
`unsupported` or `uncertain`, the next step is investigating the
claim-generation and verification path and proposing a narrow
regression test and fix, in a separate PR. If the second adjudicator
labels it `supported`, the result is an unresolved reviewer
disagreement, recorded as exactly that -- neither label is erased, and
the case is not declared resolved either way.
