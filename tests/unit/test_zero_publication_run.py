"""A run that publishes nothing, end to end.

The canonical recordings cannot demonstrate this: whether a run
publishes zero claims depends on what search returns that day, and
re-recording until one does is exactly the "rerun until it looks
right" this project refuses. So the path is driven directly instead --
a real graph run whose verifier withholds everything, rendered through
the real renderer.

What has to hold when synthesis produces nothing publishable:

* the reader still gets the sources' own words
* those words are labelled as quotations, not findings
* no generated prose appears anywhere in the fallback
* the published-claim count stays zero and excerpts are counted apart
"""

from __future__ import annotations

import pytest

from agentic_research.citations.nli import NLIPrediction, NLIScores
from agentic_research.config import LLMMode, Settings
from agentic_research.models import CitationVerification
from agentic_research.report import render_markdown
from agentic_research.runner import run_research
from fakes import FakeFetcher, FakeRouter, FakeSearchService

NOTICE = "No synthesized claim passed evidence verification"


class WithholdingScorer:
    """Entails nothing, so every claim falls below any threshold."""

    model_id = "fake/withholding"
    revision = "test"

    def score(self, pairs: list[tuple[str, str]]) -> list[NLIPrediction]:
        return [
            NLIPrediction(
                premise=p,
                hypothesis=h,
                scores=NLIScores(entailment=0.0, neutral=1.0, contradiction=0.0),
                model_id=self.model_id,
                model_revision=self.revision,
            )
            for p, h in pairs
        ]


@pytest.fixture
def withholding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real graph, real renderer, fake I/O and a verifier that entails
    nothing -- so the run reaches the fallback the way a real one would.
    """
    import agentic_research.runner as runner
    from agentic_research.graph.nodes import reporting

    monkeypatch.setattr(reporting, "build_verifier", lambda _settings: WithholdingScorer())
    monkeypatch.setattr(runner, "ModelRouter", lambda settings, tracker=None: FakeRouter())

    class _Search(FakeSearchService):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    class _Fetcher(FakeFetcher):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(runner, "build_provider", lambda settings: object())
    monkeypatch.setattr(runner, "SearchService", lambda provider, settings: _Search())
    monkeypatch.setattr(runner, "PageFetcher", lambda settings: _Fetcher())


async def zero_claim_run(tmp_path):
    settings = Settings(
        llm_mode=LLMMode.LOCAL,
        persist_runs=False,
        checkpoint_backend="none",
        output_dir=tmp_path,
    )
    return await run_research("a research question", settings)


class TestAZeroPublicationRun:
    async def test_nothing_is_published(self, withholding: None, tmp_path) -> None:
        result = await zero_claim_run(tmp_path)
        verification = CitationVerification.model_validate(result.state["verification"])
        assert verification.final_published_claims == 0
        assert verification.checked_claims > 0, "the claims were checked, not skipped"

    async def test_the_reader_still_gets_the_sources_own_words(
        self, withholding: None, tmp_path
    ) -> None:
        result = await zero_claim_run(tmp_path)
        assert NOTICE in result.markdown
        assert "## Source excerpts" in result.markdown

    async def test_the_excerpts_are_labelled_as_quotations(
        self, withholding: None, tmp_path
    ) -> None:
        result = await zero_claim_run(tmp_path)
        assert "verbatim quotations, not findings" in result.markdown
        assert "no conclusion has been drawn" in result.markdown

    async def test_every_excerpt_line_is_a_stored_quote(self, withholding: None, tmp_path) -> None:
        """The one place a system is most tempted to write something
        anyway. Every excerpt must be text the extractor stored."""
        result = await zero_claim_run(tmp_path)
        stored = {e.quote for e in result.state.get("evidence", [])}
        excerpts = [line for line in result.markdown.splitlines() if line.startswith('- "')]
        assert excerpts, "a zero-claim run rendered no excerpts"
        for line in excerpts:
            assert any(quote in line for quote in stored), f"not a stored quote: {line[:70]}"

    async def test_excerpts_are_counted_apart_from_claims(
        self, withholding: None, tmp_path
    ) -> None:
        result = await zero_claim_run(tmp_path)
        verification = CitationVerification.model_validate(result.state["verification"])
        assert verification.evidence_only_excerpts > 0
        assert verification.final_published_claims == 0

    async def test_no_claim_section_survives(self, withholding: None, tmp_path) -> None:
        """Withheld means absent, not greyed out."""
        result = await zero_claim_run(tmp_path)
        assert "## Key findings" not in result.markdown
        assert "## Summary" not in result.markdown

    async def test_the_report_still_names_its_sources(self, withholding: None, tmp_path) -> None:
        result = await zero_claim_run(tmp_path)
        assert "## Sources" in result.markdown


class TestTheFallbackCannotBeFaked:
    def test_it_does_not_appear_when_a_claim_survived(self) -> None:
        """Guards the guard: if the notice rendered unconditionally these
        tests would pass while saying nothing."""
        from agentic_research.models import Claim, ClaimKind, ResearchReport

        report = ResearchReport(
            title="T",
            key_findings=[
                Claim(
                    text="A published claim.",
                    kind=ClaimKind.FACTUAL,
                    evidence_ids=["S1-e1"],
                    citation_ids=["S1"],
                )
            ],
        )
        text = render_markdown(report, [], CitationVerification(), None, evidence=[])
        assert NOTICE not in text
