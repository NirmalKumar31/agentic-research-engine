#!/usr/bin/env python3
"""Before/after for the query and retrieval policy, spending nothing.

Every input is either a preserved artifact from a real hosted run or a
verbatim excerpt from one. Nothing here calls a provider, so it can be
re-run against the same bytes as often as needed, and its limits are
set by what those runs recorded rather than by what would be
convenient.

**What cannot be evaluated offline, stated up front.** The runs were
made with ``PERSIST_RUNS=false`` and the search stage does not stream
its candidate pool, so of 44 unique candidates only the 6 that were
selected survive -- and their provider relevance scores are not in the
serialised payload either. So this cannot answer "would the new policy
have picked better pages from the same pool": the 38 discarded
candidates are gone. It can answer what the new policy does to the
pages that *were* kept, how the two classifiers differ on the real
domains, and what the new coverage rule makes of the real extracted
quotes.

Nor can it show the new queries. Generating those needs a model call,
which is the paid run this evaluation is meant to precede.

Run: ``python examples/live-validation/tools/offline_policy_eval.py``
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from agentic_research.citations.guards import authority_of  # noqa: E402
from agentic_research.evidence.quality import (  # noqa: E402
    base_quality_for,
    classify_source,
)
from agentic_research.evidence.topicality import (  # noqa: E402
    alias_groups,
    lexically_plausible,
    salient_terms,
    shared_terms,
)
from agentic_research.graph.prompts import QUERY_WRITER_SYSTEM  # noqa: E402
from agentic_research.retrieval.selection import ranking_score  # noqa: E402

CAPTURE = REPO / "examples/live-validation/question-shapes/causes-20260930-103557"

# ---------------------------------------------------------------------------
# Verbatim from the hosted overfitting run (2026-09-30, commit b16ad010).
#
# That run was made from the browser, so no capture directory exists for
# it; these are transcribed from its own rendered report. Provenance is
# recorded here because a fixture whose origin is unstated is
# indistinguishable from one written to make a point.
# ---------------------------------------------------------------------------
OVERFITTING_SUB_QUESTIONS = [
    "How does excessive model capacity or complexity relative to the amount of "
    "training data cause overfitting?",
    "How do small, unrepresentative, or imbalanced training datasets contribute to overfitting?",
    "How do noisy features, incorrect labels, and chance correlations lead models "
    "to fit patterns that do not generalize?",
    "How can training choices such as excessive training, weak regularization, or "
    "repeated tuning cause a model to overfit?",
    "How can data leakage or repeated use of validation/test data create "
    "misleading performance estimates and conceal overfitting?",
]

OVERFITTING_DOMAINS = [
    ("x.com", "Probability and Statistics"),
    ("prachub.com", "Handle imbalance, sampling, and overfitting | LinkedIn Interview Question"),
    ("emergentmind.com", "Deep Double Descent in Neural Networks"),
    ("codefinity.com", "Capacity, Overfitting, and Generalization"),
    ("aiweeklybriefing.substack.com", "Technical Breakdown: Statistical Learning Theory"),
    ("arxiv.org", "Frozen Overparameterization: A Double Descent Perspective"),
]

OVERFITTING_EXCERPTS = [
    "the number of frozen layers can determine whether the transfer learning is "
    "effectively underparameterized or overparameterized and, in turn, this may "
    "induce a freezing-wise double descent phenomenon that determines the relative "
    "success or failure of learning.",
    "We show that the test error evolution during the target DNN training has a more "
    "significant double descent effect when the target training dataset is "
    "sufficiently large.",
    "If the capacity is too high relative to the amount of available data, the model "
    "may not only fit the underlying trend but also the random noise present in the "
    "training set.",
    "The link between VC dimension and overfitting is crucial: a hypothesis class "
    "with a VC dimension much larger than the number of training examples is likely "
    "to overfit.",
    "A high VC dimension means the hypothesis class is powerful enough to fit any "
    "possible labeling of the training data. This includes not just meaningful "
    "patterns but also random fluctuations and noise.",
    "When a model memorizes noise, it loses its ability to generalize well to new, "
    "unseen data, as the learned patterns do not reflect the underlying data "
    "distribution.",
    "Regularization techniques, cross-validation, and model selection strategies can "
    "help prevent overfitting by limiting capacity or penalizing overly complex "
    "models.",
    "Interpolation threshold: As capacity reaches the number of data points, the "
    "system attains zero training error. Label noise or residuals in the data matrix "
    "give rise to small singular directions, leading to a variance spike in "
    "predictions.",
    "Overparameterized networks learn to allocate parameters to noise directions with "
    "low norm, allowing 'benign overfitting'.",
    "Sufficiently strong weight decay suppresses excess parameter variance at "
    "interpolation, resulting in a monotonic generalization curve.",
]

# The classifier as it was before SOCIAL and REFERENCE existed: every
# unrecognised domain fell through to `other`.
_PRE_CHANGE_UNRECOGNISED = {
    "x.com",
    "twitter.com",
    "linkedin.com",
    "prachub.com",
    "wikipedia.org",
    "en.wikipedia.org",
    "youtube.com",
}


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def recorded_events() -> list[dict]:
    raw = json.loads((CAPTURE / "events.json").read_text())
    out: list[dict] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if "event" in node and not isinstance(node.get("data"), dict):
                out.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(raw)
    return out


def section_queries() -> None:
    rule("1. QUERIES — recorded verbatim from the hallucination run")
    events = recorded_events()
    queries: list[str] = next(
        (e["queries"] for e in events if e.get("event") == "queries_generated"), []
    )
    print(f"{len(queries)} queries issued. Word counts:\n")
    for i, query in enumerate(queries, 1):
        print(f"  Q{i} ({len(query.split()):2d} words) {query}")
    counts = sorted(len(q.split()) for q in queries)
    median = (
        counts[len(counts) // 2]
        if len(counts) % 2
        else (counts[len(counts) // 2 - 1] + counts[len(counts) // 2]) / 2
    )
    print(f"\n  median words: {median}   min: {counts[0]}   max: {counts[-1]}")
    print("  the new prompt asks for three to eight words")

    print("\nWhat changed in the prompt (inspectable, no model needed):")
    removed = "specific technical terms an authoritative page"
    print(f"  instruction removed : {removed!r} -> {removed not in QUERY_WRITER_SYSTEM}")
    for phrase, label in [
        ("own vocabulary", "asks for the question's own words"),
        ("three to eight words", "bounds length"),
        ("proper name", "protects names from that bound"),
        ("version number", "protects versions"),
        ("overfitting", "carries the failure that motivated it"),
    ]:
        print(f"  {label:38} -> {phrase in QUERY_WRITER_SYSTEM}")
    print("\n  NOT EVALUABLE OFFLINE: the queries the new prompt produces.")
    print("  That needs a model call, which is the paid run this precedes.")


def section_classification() -> None:
    rule("2. SOURCE CLASSIFICATION — before/after on the real domains")
    print(f"{'domain':32} {'was':10} {'now':10} {'quality':>8}  authority")
    print("-" * 74)
    domains = [d for d, _ in OVERFITTING_DOMAINS] + [
        "vstorm.co",
        "atlan.com",
        "mdpi.com",
        "en.wikipedia.org",
    ]
    for domain in dict.fromkeys(domains):
        url = f"https://{domain}/page"
        now = classify_source(url, domain)
        was = "other" if domain in _PRE_CHANGE_UNRECOGNISED else now.value
        flag = "  <-- changed" if was != now.value else ""
        print(
            f"{domain:32} {was:10} {now.value:10} "
            f"{base_quality_for(now):>8.2f}  {authority_of(now.value).value}{flag}"
        )


def section_selection() -> None:
    rule("3. SELECTION — new ranking on the real domain set")
    print("ASSUMPTION, stated: provider relevance scores are not in the")
    print("serialised payload, so every candidate is scored equally at 0.80.")
    print("That isolates the policy change; it is not the real pool.\n")
    print(f"{'domain':32} {'class':10} {'old key':>9} {'new score':>10}")
    print("-" * 74)

    class Fake:
        def __init__(self, domain: str) -> None:
            self.domain = domain
            self.url = f"https://{domain}/page"
            self.best_score = 0.80

    rows = []
    for domain, _ in OVERFITTING_DOMAINS:
        candidate = Fake(domain)
        source_type = classify_source(candidate.url, domain)
        new = ranking_score(candidate, explanatory=True)  # type: ignore[arg-type]
        rows.append((domain, source_type.value, 0.8, new))
    for domain, cls, old, new in sorted(rows, key=lambda r: -r[3]):
        print(f"{domain:32} {cls:10} {old:>9.1f} {new:>10.2f}")
    print("\nOld key banded relevance to tenths and used authority only to break")
    print("ties inside a band, so at equal scores the order was the URL. Six of")
    print("six candidates sat in one band, which is how a tweet was read.")
    top3 = [r[0] for r in sorted(rows, key=lambda r: -r[3])[:3]]
    print(f"\nNew top three: {top3}")
    ordered = [r[0] for r in sorted(rows, key=lambda r: -r[3])]
    print(f"x.com now ranks: {ordered.index('x.com') + 1} of {len(rows)}")


def section_coverage() -> None:
    rule("4. COVERAGE — old rule vs new, on the real extracted quotes")
    print("Old rule: an exact-match quote counted toward the sub-question it was")
    print("filed under, whatever it was about. New rule: it must also discuss")
    print("that sub-question's own terms.")
    print()
    print("The run's actual quote-to-sub-question attribution was not recorded,")
    print("so this measures every quote against every sub-question: how many of")
    print("the 10 preserved excerpts each sub-question would admit.")
    print()

    question = "What are the main causes of overfitting in machine learning?"
    aliases = alias_groups(question, ["overfitting", "machine learning"])
    topic = salient_terms(question)

    print(
        f"{'sub-question':<6} {'old':>4} {'new':>4}  {'admitted quotes':<16} discriminating terms"
    )
    print("-" * 74)
    total_old = total_new = 0
    for i, sub_question in enumerate(OVERFITTING_SUB_QUESTIONS, 1):
        admitted = [
            q
            for q in OVERFITTING_EXCERPTS
            if lexically_plausible(sub_question, q, aliases=aliases, topic_terms=topic)
        ]
        total_old += len(OVERFITTING_EXCERPTS)
        total_new += len(admitted)
        best = max(
            (shared_terms(sub_question, q, aliases=aliases) - topic for q in OVERFITTING_EXCERPTS),
            key=len,
            default=frozenset(),
        )
        print(
            f"SQ{i:<5} {len(OVERFITTING_EXCERPTS):>4} {len(admitted):>4}  "
            f"{'#' * len(admitted):<16} {sorted(best)[:5]}"
        )
    print("-" * 74)
    print(f"{'total':<6} {total_old:>4} {total_new:>4}")
    refused = total_old - total_new
    print(
        f"\n  The new rule refuses {refused} of {total_old} quote/sub-question pairings"
        f" ({refused / total_old:.0%})."
    )
    print("  The old rule refused none: an exact quote counted wherever it was filed.")
    print()
    print("  MEASURED LIMITS, not claims:")
    print("   - the double-descent excerpts share nothing with the capacity or")
    print("     leakage sub-questions and are refused, which is the false positive")
    print("     that mattered;")
    print("   - SQ5 (data leakage) still admits a quote about cross-validation as a")
    print("     prevention technique, because both say 'validation' in different")
    print("     senses. A lexical bound cannot separate those, and this one does")
    print("     not pretend to: the consequence is SQ5 reported 'weak' rather than")
    print("     'uncovered', which is a smaller error than counting it covered.")


def section_sse() -> None:
    rule("5. SSE LIFECYCLE — the duplicate start, in the recorded baseline")
    raw = json.loads((CAPTURE / "events.json").read_text())
    # Both the transport-level event and the runner's, which the web
    # layer forwarded as progress. Counting only the outer envelope
    # misses the duplicate, which is how it survived.
    outer = sum(1 for e in raw if e.get("event") == "started")
    nested = sum(
        1 for e in raw if isinstance(e.get("data"), dict) and e["data"].get("event") == "started"
    )
    print(f"  transport 'started' events      : {outer}")
    print(f"  runner 'started' forwarded      : {nested}")
    print(f"  visible to the page in baseline : {outer + nested}")
    print("  after the fix: exactly one, asserted by test_web_api against a")
    print("  stream that injects the runner's own start event")


def section_limits() -> None:
    rule("6. WHAT THIS EVALUATION CANNOT SHOW")
    for line in [
        "The 38 discarded candidates. PERSIST_RUNS=false and the search stage",
        "  does not stream its pool, so only the 6 selected pages survive. Whether",
        "  the new policy would have chosen better pages FROM THE SAME POOL is",
        "  therefore unanswerable offline -- the pool is gone.",
        "",
        "Provider relevance scores. Not in the serialised payload, so section 3",
        "  assumes equal relevance. A real pool has a spread, and the cap on the",
        "  authority adjustment (0.35) means a wide spread still decides.",
        "",
        "The queries the new prompt produces. Needs a model call.",
        "",
        "Whether Tavily returns suitable pages for a plainer query. This is the",
        "  open question, and it is the one the paid run exists to answer. If the",
        "  candidate pool genuinely lacks good explanatory sources for this",
        "  question, no prompt fix can manufacture them.",
    ]:
        print(f"  {line}")


def main() -> int:
    print("OFFLINE POLICY EVALUATION — no provider calls, no credentials")
    print(f"capture: {CAPTURE.relative_to(REPO)}")
    section_queries()
    section_classification()
    section_selection()
    section_coverage()
    section_sse()
    section_limits()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
