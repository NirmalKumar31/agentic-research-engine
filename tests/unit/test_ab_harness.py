"""Controlled model comparison over a frozen evidence corpus.

The harness exists so a model comparison is not confounded by search
results moving between runs. These tests pin the properties that make the
comparison valid: the corpus round-trips unchanged, replay issues no
searches, and both arms see identical input.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentic_research.config import ModelRole, Settings
from agentic_research.evaluation.ab import EvidenceCorpus, all_roles, compare, run_arm
from agentic_research.models import (
    DiscoveryRef,
    EvidenceItem,
    QuoteMatch,
    SearchQuery,
    SourceDocument,
    SubQuestion,
)
from fakes import FakeRouter

PAGE = "Fraud datasets are severely imbalanced in production systems everywhere."


def corpus() -> EvidenceCorpus:
    source = SourceDocument(
        id="S1", url="https://x.org/a", canonical_url="https://x.org/a", title="Paper",
        domain="x.org", text=PAGE, content_hash="h",
        discovered_by=[DiscoveryRef(query_id="Q1", sub_question_id="SQ1")],
    )
    item = EvidenceItem(
        id="S1-e1", source_id="S1", sub_question_id="SQ1",
        claim="Fraud datasets are imbalanced.",
        quote="Fraud datasets are severely imbalanced in production systems",
        quote_match=QuoteMatch.EXACT_NORMALIZED, relevance=0.9,
        discovery=DiscoveryRef(query_id="Q1", sub_question_id="SQ1"),
    )
    return EvidenceCorpus(
        question="Is fraud data imbalanced?",
        sub_questions=[SubQuestion(id="SQ1", text="how imbalanced?", rationale="r")],
        sources=[source],
        evidence=[item],
        completed_queries=[
            SearchQuery(id="Q1", sub_question_id="SQ1", text="q", round_number=1)
        ],
    )


@pytest.fixture
def ab_settings(settings: Settings) -> Settings:
    settings.max_research_rounds = 1
    return settings


@pytest.fixture(autouse=True)
def _fake_models(monkeypatch: pytest.MonkeyPatch) -> None:
    import agentic_research.evaluation.ab as ab

    monkeypatch.setattr(ab, "ModelRouter", lambda settings, tracker=None: FakeRouter())


class TestCorpusRoundTrip:
    def test_corpus_survives_serialisation_unchanged(self, tmp_path: Path) -> None:
        """Both arms must see identical input; a lossy round trip would
        quietly reintroduce the variance the harness removes."""
        original = corpus()
        restored = EvidenceCorpus.load(original.save(tmp_path / "c.json"))

        assert restored.question == original.question
        assert [s.id for s in restored.sources] == [s.id for s in original.sources]
        assert [e.id for e in restored.evidence] == [e.id for e in original.evidence]
        assert restored.evidence[0].quote == original.evidence[0].quote
        assert restored.evidence[0].quote_match is QuoteMatch.EXACT_NORMALIZED
        # Provenance must survive too, or replay cannot verify citations.
        assert restored.evidence[0].discovery == original.evidence[0].discovery
        assert restored.sources[0].discovered_by == original.sources[0].discovered_by

    def test_summary_reports_citable_count(self) -> None:
        assert "1 citable" in corpus().summary()


class TestReplay:
    async def test_replay_produces_a_verified_report(self, ab_settings: Settings) -> None:
        result = await run_arm("local", corpus(), ab_settings)
        assert result.ok, result.error
        assert result.markdown.startswith("# ")
        assert result.llm_calls > 0
        assert result.value("citation_integrity") == 1.0

    async def test_replay_never_searches(self, ab_settings: Settings) -> None:
        """If replay searched, the two arms would see different evidence and
        the comparison would be meaningless. The stub raises rather than
        returning empty, so a regression fails loudly."""
        result = await run_arm("local", corpus(), ab_settings)
        assert result.ok, result.error

    async def test_verification_is_exhaustive_in_replay(
        self, ab_settings: Settings
    ) -> None:
        result = await run_arm("local", corpus(), ab_settings)
        assert result.verification.get("entailment_exhaustive") is True


class TestComparison:
    async def test_both_arms_run_over_the_same_corpus(
        self, ab_settings: Settings
    ) -> None:
        shared = corpus()
        comparison = await compare(
            shared,
            ab_settings,
            {
                "local": all_roles("ollama:qwen3:4b"),
                "cloud-cheap": all_roles("ollama:llama3.2:3b"),
            },
        )
        assert len(comparison.arms) == 2
        assert all(a.ok for a in comparison.arms)
        assert comparison.corpus_summary == shared.summary()

    async def test_report_records_that_the_corpus_was_shared(
        self, ab_settings: Settings
    ) -> None:
        comparison = await compare(
            corpus(), ab_settings, {"a": all_roles("ollama:qwen3:4b")}
        )
        payload = comparison.to_dict()
        assert "frozen evidence corpus" in payload["note"]
        assert payload["environment"]["python"]
        assert payload["arms"][0]["metrics"]

    async def test_a_failing_arm_does_not_lose_the_others(
        self, ab_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import agentic_research.evaluation.ab as ab

        calls = {"n": 0}
        real = ab.run_arm

        async def flaky(label, corpus_, settings, **kwargs):  # noqa: ANN001, ANN003
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("provider down")
            return await real(label, corpus_, settings, **kwargs)

        # compare() calls run_arm directly, so patch where it is looked up.
        monkeypatch.setattr(ab, "run_arm", flaky)
        with pytest.raises(RuntimeError):
            await ab.compare(corpus(), ab_settings, {"a": {}, "b": {}})

    def test_all_roles_rejects_a_malformed_spec(self) -> None:
        with pytest.raises(ValueError):
            all_roles("not-a-spec")

    def test_all_roles_covers_every_role(self) -> None:
        assert set(all_roles("ollama:qwen3:4b")) == set(ModelRole)
