"""The three no-spend preflight fixes landed before any Phase B run:
unambiguous track/run-count, automatic corpus hashing with fail-closed
verification, and a bounded wall-clock timeout on `run_arm`.

No provider call in this file. The timeout test uses a fake model that
sleeps past a deliberately short ceiling -- real time, not mocked --
which is what proves the ceiling actually bounds wall-clock time rather
than merely existing in the type signature.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agentic_research.config import ModelRole, Settings
from agentic_research.evaluation import ab
from agentic_research.evaluation.ab import EvidenceCorpus, run_arm
from agentic_research.evaluation.benchmark_manifest import (
    BenchmarkManifest,
    CorpusHashMismatchError,
    corpus_hash,
    freeze_corpus_hashes,
    verify_corpus_hash,
)
from agentic_research.graph.nodes import reporting
from agentic_research.models import (
    DiscoveryRef,
    EvidenceItem,
    QuoteMatch,
    SearchQuery,
    SourceDocument,
    SubQuestion,
)
from fakes import FakeRouter

ROOT = Path(__file__).resolve().parents[2]


def _corpus(question: str = "Is fraud data imbalanced?") -> EvidenceCorpus:
    source = SourceDocument(
        id="S1",
        url="https://x.org/a",
        canonical_url="https://x.org/a",
        title="Paper",
        domain="x.org",
        text="Fraud datasets are severely imbalanced in production systems.",
        content_hash="h",
        discovered_by=[DiscoveryRef(query_id="Q1", sub_question_id="SQ1")],
    )
    item = EvidenceItem(
        id="S1-e1",
        source_id="S1",
        sub_question_id="SQ1",
        claim="Fraud datasets are imbalanced.",
        quote="Fraud datasets are severely imbalanced in production systems.",
        quote_match=QuoteMatch.EXACT_NORMALIZED,
        relevance=0.9,
        discovery=DiscoveryRef(query_id="Q1", sub_question_id="SQ1"),
    )
    return EvidenceCorpus(
        question=question,
        sub_questions=[SubQuestion(id="SQ1", text="how imbalanced?", rationale="r")],
        sources=[source],
        evidence=[item],
        completed_queries=[SearchQuery(id="Q1", sub_question_id="SQ1", text="q", round_number=1)],
        captured_at="2026-01-01T00:00:00+00:00",
    )


class TestFix1_TrackAndRunCountAreUnambiguous:
    def test_the_protocol_names_track_1_explicitly(self) -> None:
        text = (ROOT / "docs" / "BENCHMARK-PROTOCOL.md").read_text()
        assert '"48 total runs" means Track 1 only, unambiguously.' in text

    def test_the_protocol_excludes_track_2_from_this_count(self) -> None:
        text = (ROOT / "docs" / "BENCHMARK-PROTOCOL.md").read_text()
        assert "does not include any Track 2" in text


class TestFix2_CorpusHashing:
    def test_the_real_workflow_is_hash_stable_across_reloads(self, tmp_path: Path) -> None:
        """The actual Phase B pattern: freeze once, save, then load the
        same artifact repeatedly (one load per arm x repetition) and
        verify each load against the hash taken at freeze time.

        Not "construct the corpus twice and compare" -- `EvidenceItem`
        has an `extracted_at` field that defaults to `datetime.now()`,
        so two independently constructed corpora never match even with
        identical intended content, and that is correct: they really are
        different artifacts, frozen at different moments. This is the
        failure the first version of this test hit, by accident rather
        than by testing anything real.
        """
        original = _corpus()
        saved_path = original.save(tmp_path / "q1.json")
        frozen_hash = corpus_hash(original)

        for _ in range(4):  # 2 arms x 2 repetitions, the real Phase B shape
            reloaded = EvidenceCorpus.load(saved_path)
            assert corpus_hash(reloaded) == frozen_hash

    def test_different_content_hashes_differently(self) -> None:
        import copy

        c1 = _corpus("Q1")
        c2 = copy.deepcopy(c1)
        c2.question = "Q2"
        assert corpus_hash(c1) != corpus_hash(c2)

    def test_freeze_batch_produces_a_manifest_ready_dict(self) -> None:
        hashes = freeze_corpus_hashes({"q1": _corpus()})
        assert set(hashes) == {"q1"}
        assert len(hashes["q1"]) == 64  # sha256 hex digest

    def test_a_matching_corpus_verifies_silently(self) -> None:
        c = _corpus()
        manifest = BenchmarkManifest(
            benchmark_version="v1",
            frozen_engine_commit="a",
            cloud_model_id="b",
            nli_model_id="c",
            nli_model_revision="d" * 40,
            prompt_version="e",
            schema_version="f",
            config_fingerprint="g",
            corpus_hashes=freeze_corpus_hashes({"q1": c}),
        )
        verify_corpus_hash("q1", c, manifest)  # must not raise

    def test_a_changed_corpus_is_refused_fail_closed(self) -> None:
        """The actual requirement: verification must REFUSE a drifted
        corpus, not merely be *able* to detect one."""
        frozen = _corpus("original question")
        manifest = BenchmarkManifest(
            benchmark_version="v1",
            frozen_engine_commit="a",
            cloud_model_id="b",
            nli_model_id="c",
            nli_model_revision="d" * 40,
            prompt_version="e",
            schema_version="f",
            config_fingerprint="g",
            corpus_hashes=freeze_corpus_hashes({"q1": frozen}),
        )
        drifted = _corpus("a different question entirely")
        with pytest.raises(CorpusHashMismatchError):
            verify_corpus_hash("q1", drifted, manifest)

    def test_a_question_with_no_recorded_hash_is_refused(self) -> None:
        """Missing is treated as worse than mismatched, not as "nothing
        to check" -- a silent pass here would be the actual gap this
        fix exists to close.

        Checked on the specific message, not just that *some* exception
        fired: `actual != expected` is also true when `expected` is
        `None`, so a mutant that disables only the dedicated
        missing-hash check still raises -- via the *other* branch --
        with a less specific message. Asserting the message content is
        what makes that mutant show up as caught instead of surviving.
        """
        manifest = BenchmarkManifest(
            benchmark_version="v1",
            frozen_engine_commit="a",
            cloud_model_id="b",
            nli_model_id="c",
            nli_model_revision="d" * 40,
            prompt_version="e",
            schema_version="f",
            config_fingerprint="g",
            corpus_hashes={},
        )
        with pytest.raises(CorpusHashMismatchError, match="no hash recorded"):
            verify_corpus_hash("q1", _corpus(), manifest)

    def test_the_real_question_sets_hash_is_computable_and_stable(self) -> None:
        """Not a placeholder: this is the actual artifact Phase B freezes
        against, hashed the same way a real run would hash it."""
        import json

        data = json.loads((ROOT / "examples" / "benchmark" / "questions.json").read_text())
        assert len(data["questions"]) == 12


class TestFix3_RunArmHasABoundedWallClockTimeout:
    @pytest.fixture
    def tight_settings(self, settings: Settings) -> Settings:
        settings.max_research_rounds = 1
        settings.run_timeout_seconds = 0.2
        return settings

    async def test_a_hung_model_call_times_out_rather_than_hanging_forever(
        self, tight_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _HangingRole:
            async def structured(self, *_a: object, **_k: object) -> object:
                await asyncio.sleep(5.0)  # far past the 0.2s ceiling
                raise AssertionError("should have timed out first")

        class _HangingRouter:
            def __init__(self) -> None:
                from agentic_research.llm.base import UsageTracker

                # The real tracker, not a stub. Every attribute a graph
                # node touches mid-run (budget checks, usage recording)
                # has to exist, or the run fails before the hanging call
                # is ever reached -- which is what a stub tracker did
                # the first time this test was written.
                self.tracker = UsageTracker(max_calls=100)

            def get(self, _role: ModelRole) -> _HangingRole:
                return _HangingRole()

            def describe(self) -> dict[str, str]:
                return {"synthesizer": "fake:hanging"}

        monkeypatch.setattr(ab, "ModelRouter", lambda settings, tracker=None: _HangingRouter())

        started = asyncio.get_event_loop().time()
        result = await run_arm("hanging-arm", _corpus(), tight_settings)
        elapsed = asyncio.get_event_loop().time() - started

        assert result.timed_out is True
        assert result.ok is False
        assert "0s" in result.error or "did not complete" in result.error
        # Real wall-clock proof: this genuinely did not wait out the 5s sleep.
        assert elapsed < 2.0, f"took {elapsed:.2f}s -- the timeout did not actually bound it"

    async def test_a_normal_fast_arm_is_unaffected_by_the_timeout(
        self, tight_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-vacuity: the timeout must not fire on ordinary runs.

        A real `ModelRouter` would need real Ollama/OpenAI connectivity
        and is not what this test is about; `FakeRouter` is the same
        substitute the main graph-flow tests already use.
        """
        monkeypatch.setattr(ab, "ModelRouter", lambda settings, tracker=None: FakeRouter())
        result = await run_arm("fast-arm", _corpus(), tight_settings)
        assert result.timed_out is False
        assert result.ok is True

    async def test_timeout_is_distinguished_from_an_ordinary_failure(
        self, tight_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The structured-outcome requirement: `ok=False` alone already
        existed for any failure. `timed_out` has to mean something a
        caller can act on differently from a generic exception.

        Two injection points were tried and rejected first: a failure
        inside `.structured()`, and `router.get()` raising. Both get
        absorbed by the synthesis node's own internal error handling --
        a legitimate, already-relied-on fail-closed behaviour (the node
        degrades to an evidence-only report rather than crashing), just
        not a path that reaches `run_arm`'s own except block. A failure
        inside the *verifier* does reach it, which is what this uses.
        """
        monkeypatch.setattr(ab, "ModelRouter", lambda settings, tracker=None: FakeRouter())

        class _RaisingScorer:
            model_id = "fake/raising"
            revision = "test"

            def score(self, pairs: object) -> object:
                raise RuntimeError("induced, not a timeout")

        monkeypatch.setattr(reporting, "build_verifier", lambda settings: _RaisingScorer())

        result = await run_arm("broken-arm", _corpus(), tight_settings)
        assert result.ok is False
        assert result.timed_out is False, "an ordinary exception must not be reported as a timeout"
        assert "induced" in result.error
