"""What a report may say when a provider fails mid-run.

Both behaviours here were defects in one real live run. The provider
returned 429 during synthesis and again during verification, and the
published report carried this in its limitations:

    "Rate limit reached for gpt-6-luna in organization org-44ebai..."

which put the account's organisation id in front of every visitor. The
same run then published zero claims: the degraded evidence listing that
replaces a failed report was gated on entailment verdicts that the
outage had also prevented, so every finding the run had paid to retrieve
was discarded.
"""

from __future__ import annotations

import pytest

from agentic_research.evidence.store import EvidenceStore
from agentic_research.graph.nodes.reporting import _fallback_report, _safe_failure_reason
from agentic_research.llm.base import (
    CloudBudgetExceededError,
    LLMError,
    ModelTimeoutError,
    ModelUnavailableError,
    ProviderRateLimited,
    ProviderRejectedRequest,
    StructuredOutputError,
)
from agentic_research.models import (
    ClaimKind,
    ContentOrigin,
    EvidenceItem,
    QuoteMatch,
    SourceDocument,
)

# Shaped like the body OpenAI actually returned, including the account id.
RAW_PROVIDER_ERROR = (
    "openai:gpt-6-luna rate limited or out of quota: Error code: 429 - "
    "{'error': {'message': 'Rate limit reached for gpt-6-luna in organization "
    "org-44ebaiE0qDzPjyakgAqwDGql on tokens per min (TPM): Limit 10000, Used 9800'}}"
)
ORG_ID = "org-44ebaiE0qDzPjyakgAqwDGql"


def _store(count: int = 3) -> EvidenceStore:
    source = SourceDocument(
        id="S1",
        url="https://example.org/a",
        canonical_url="https://example.org/a",
        title="A source",
        domain="example.org",
        text="body",
        content_hash="h",
        content_origin=ContentOrigin.HTML_FETCH,
    )
    evidence = [
        EvidenceItem(
            id=f"S1-e{i}",
            source_id="S1",
            sub_question_id="SQ1",
            claim=f"An extracted finding {i}.",
            quote=f"quote {i}",
            quote_match=QuoteMatch.EXACT_NORMALIZED,
            relevance=0.9,
        )
        for i in range(count)
    ]
    return EvidenceStore([source], evidence)


class TestNoProviderDetailIsPublished:
    @pytest.mark.parametrize(
        "exc",
        [
            ProviderRateLimited(RAW_PROVIDER_ERROR),
            CloudBudgetExceededError("output_tokens", 19_500, 20_000),
            ModelTimeoutError("timed out talking to openai:gpt-6-luna"),
            ModelUnavailableError.__new__(ModelUnavailableError),
            StructuredOutputError.__new__(StructuredOutputError),
            ProviderRejectedRequest("400 from openai:gpt-6-luna"),
            LLMError("something else entirely"),
        ],
    )
    def test_the_reason_never_echoes_the_exception(self, exc: BaseException) -> None:
        reason = _safe_failure_reason(exc)
        assert reason
        assert ORG_ID not in reason
        assert "429" not in reason
        assert "{" not in reason

    def test_the_organisation_id_cannot_reach_the_report(self) -> None:
        report = _fallback_report(
            "A question?",
            _store(),
            [],
            _safe_failure_reason(ProviderRateLimited(RAW_PROVIDER_ERROR)),
        )
        published = " ".join(report.limitations)
        assert ORG_ID not in published
        assert "org-" not in published
        assert "rate limit was reached" in published.lower()

    def test_no_configured_setting_name_is_published(self) -> None:
        """A budget message naming MAX_CLOUD_OUTPUT_TOKENS tells a visitor
        how the deployment is configured."""
        reason = _safe_failure_reason(CloudBudgetExceededError("output_tokens", 19_500, 20_000))
        assert "MAX_" not in reason
        assert "19500" not in reason


class TestTheDegradedListingSurvives:
    """The fallback exists so a run that gathered findings does not return
    nothing when the final call fails. Gating it on entailment defeated
    exactly that, in the one situation it was built for."""

    def test_extracted_findings_are_published_without_a_verdict(self) -> None:
        from agentic_research.citations.publication import filter_report_by_verification

        report = _fallback_report("A question?", _store(3), [], "the provider failed")
        assert len(report.substantive_claims()) == 3

        # No verdicts at all: verification never ran.
        filtered, removed = filter_report_by_verification(report, {})

        assert removed == 0
        assert len(filtered.substantive_claims()) == 3

    def test_they_are_marked_as_extracted_not_factual(self) -> None:
        report = _fallback_report("A question?", _store(2), [], "the provider failed")
        assert {c.kind for c in report.substantive_claims()} == {ClaimKind.EXTRACTED}

    def test_each_one_still_carries_its_evidence(self) -> None:
        report = _fallback_report("A question?", _store(3), [], "the provider failed")
        for claim in report.substantive_claims():
            assert len(claim.evidence_ids) == 1

    def test_the_report_states_what_was_and_was_not_checked(self) -> None:
        report = _fallback_report("A question?", _store(2), [], "the provider failed")
        framing = " ".join(c.text for c in report.summary_claims)
        assert "reproduced exactly from its source" in framing
        assert "no claim has been entailment-checked" in framing

    def test_the_published_text_is_the_quote_not_the_paraphrase(self) -> None:
        """EXTRACTED bypasses entailment, so what it publishes must carry
        a guarantee of its own. Exact quote matching proves the *quote*
        appears in the source and says nothing about whether the
        extractor's paraphrase beside it is faithful."""
        store = _store(1)
        item = store.citable_evidence()[0]
        report = _fallback_report("A question?", store, [], "the provider failed")

        published = [c.text for c in report.key_findings]
        assert published == [item.quote]
        assert item.claim not in published

    def test_only_citable_evidence_is_excerpted(self) -> None:
        """A fuzzy or unmatched quote was never aligned to its source, so
        it carries no guarantee at all."""
        from agentic_research.evidence.store import EvidenceStore

        store = _store(2)
        fuzzy = EvidenceItem(
            id="S1-e99",
            source_id="S1",
            sub_question_id="SQ1",
            claim="unaligned",
            quote="a span that was never found in the page",
            quote_match=QuoteMatch.FUZZY,
            relevance=0.99,
        )
        widened = EvidenceStore(list(store.sources), [*store.citable_evidence(), fuzzy])
        report = _fallback_report("A question?", widened, [], "the provider failed")

        assert fuzzy.quote not in [c.text for c in report.key_findings]

    def test_the_fallback_needs_no_model_call(self) -> None:
        """It runs when the provider is unavailable, so it cannot depend
        on one."""
        import inspect

        from agentic_research.graph.nodes import reporting

        body = inspect.getsource(reporting._fallback_report)
        assert "structured(" not in body
        assert "router" not in body

    def test_a_synthesised_claim_is_still_gated(self) -> None:
        """The exemption must not become a way to publish unverified
        synthesis: only the extracted kind is exempt."""
        from agentic_research.citations.publication import filter_report_by_verification
        from agentic_research.models import Claim, ResearchReport

        synthesised = Claim(
            text="A conclusion drawn across sources.",
            evidence_ids=["S1-e0"],
            kind=ClaimKind.SYNTHESIS,
        )
        report = ResearchReport(title="T", summary_claims=[synthesised])
        filtered, removed = filter_report_by_verification(report, {})

        assert removed == 1
        assert filtered.substantive_claims() == []
