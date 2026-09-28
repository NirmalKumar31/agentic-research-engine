"""What a run has to leave behind for someone to check it later.

v1.2.0 added four decisions to the verification path: the contract the
question was decomposed into, the propositions a claim was split into,
the relevance gate, and the bounded wording repair. Every one of them
was reaching the reader only as prose inside a ``reason`` string, and
two were not reaching it at all -- the contract was discarded with the
state, and ``claim.text`` is reassigned in place when a repair is
accepted, so the wording a published claim used to have existed
nowhere.

That is the same defect as the 200-character truncation that left four
claims in the canonical recordings ending mid-sentence: the only copy
of something was lossy, and by the time anyone wanted it the run was
over. A hosted acceptance capture is exactly "by the time anyone
wanted it", so these assertions are about what survives the run rather
than about what the gates decide.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import pytest

from agentic_research.answer_contract import QuestionType, build_contract
from agentic_research.citations.nli import NLIPrediction, NLIScores
from agentic_research.citations.semantic import CitedEvidence, verify_claim
from agentic_research.config import LLMMode, Settings
from agentic_research.evidence.store import EvidenceStore
from agentic_research.graph.nodes import reporting
from agentic_research.models import Claim, ClaimJudgment, ClaimKind
from agentic_research.runner import run_research
from agentic_research.schemas import RepairOut, RewriteOut
from agentic_research.web.recordings import serialise_result
from fakes import FakeFetcher, FakeRouter, FakeSearchService

THRESHOLD = 0.98


class ScriptedScorer:
    """Entails a hypothesis when the script says so, and not otherwise.

    Keyed on a substring of the hypothesis rather than on call order:
    verify_claim recurses per proposition and then re-scores the whole
    claim, so an order-keyed fake would silently answer the wrong
    question the moment the recursion changed.
    """

    model_id = "fake/scripted"
    revision = "test"

    def __init__(self, entailed: dict[str, float], default: float = 0.0) -> None:
        self._entailed = entailed
        self._default = default
        self.seen: list[str] = []

    def _score_for(self, hypothesis: str) -> float:
        for needle, value in self._entailed.items():
            if needle.lower() in hypothesis.lower():
                return value
        return self._default

    def score(self, pairs: list[tuple[str, str]]) -> list[NLIPrediction]:
        out = []
        for premise, hypothesis in pairs:
            self.seen.append(hypothesis)
            entail = self._score_for(hypothesis)
            out.append(
                NLIPrediction(
                    premise=premise,
                    hypothesis=hypothesis,
                    scores=NLIScores(
                        entailment=entail,
                        neutral=round(1.0 - entail, 6),
                        contradiction=0.0,
                    ),
                    model_id=self.model_id,
                    model_revision=self.revision,
                )
            )
        return out


# A comma before the conjunction plus a new subject with a finite verb:
# the decomposer splits on that, not on the word "and" alone.
BUNDLED = "SMOTE oversamples the minority class, and the model recall improved substantially."
QUOTE = "SMOTE oversamples the minority class."


def pairs(quote: str = QUOTE) -> list[CitedEvidence]:
    return [CitedEvidence(evidence_id="S1-e1", quote=quote)]


# Repair runs behind every earlier gate, so the fixture has to clear
# them all or a test about the repair record fails somewhere else. A
# definition question is the simplest shape that does: one subject, so
# the atomicity guard passes, and one entity, so the relevance gate is
# satisfied without a contrast.
DEFINITION_QUOTE = "SMOTE oversamples the minority class."
OVERSTATED = "SMOTE always oversamples the minority class."
WEAKENED = "SMOTE oversamples the minority class."


class TestThePropositionsSurviveTheVerdict:
    """Which half of a bundled claim failed, and by how far."""

    def test_a_split_claim_records_every_part_it_checked(self) -> None:
        scorer = ScriptedScorer({"oversamples": 0.99, "recall improved": 0.10})
        verdict = verify_claim(BUNDLED, pairs(), scorer, support_threshold=THRESHOLD)

        assert not verdict.publishable
        texts = [p.text for p in verdict.propositions]
        assert len(texts) == 2, f"expected both halves, got {texts}"
        assert [p.supported for p in verdict.propositions] == [True, False]

    def test_the_failing_part_carries_its_own_score(self) -> None:
        """The number that used to be unrecoverable. "One of these two
        is unsupported" is a different finding from "it missed by
        0.88"."""
        scorer = ScriptedScorer({"oversamples": 0.99, "recall improved": 0.10})
        verdict = verify_claim(BUNDLED, pairs(), scorer, support_threshold=THRESHOLD)

        failed = [p for p in verdict.propositions if not p.supported]
        assert len(failed) == 1
        assert failed[0].best_entailment == pytest.approx(0.10, abs=1e-6)

    def test_a_claim_that_publishes_still_records_its_parts(self) -> None:
        """Not only failures. A reader checking a published claim needs
        to see that each of its assertions was carried on its own."""
        scorer = ScriptedScorer({"oversamples": 0.99, "recall improved": 0.99})
        verdict = verify_claim(BUNDLED, pairs(), scorer, support_threshold=THRESHOLD)

        assert verdict.publishable
        assert len(verdict.propositions) == 2
        assert all(p.supported for p in verdict.propositions)

    def test_an_atomic_claim_records_none(self) -> None:
        """Non-vacuity: the list is populated by decomposition, not by
        every call. A single assertion was never split, and reporting
        one proposition would misdescribe what happened."""
        scorer = ScriptedScorer({"oversamples": 0.99})
        verdict = verify_claim(
            "SMOTE oversamples the minority class.", pairs(), scorer, support_threshold=THRESHOLD
        )
        assert verdict.publishable
        assert verdict.propositions == []

    def test_scoring_stops_at_the_first_failure(self) -> None:
        """Documents the one place the record is deliberately partial.

        A failed proposition withholds the claim, so the parts after it
        are never scored and must not be reported as though they were.
        """
        scorer = ScriptedScorer({"oversamples": 0.05, "recall improved": 0.99})
        verdict = verify_claim(BUNDLED, pairs(), scorer, support_threshold=THRESHOLD)

        assert not verdict.publishable
        assert len(verdict.propositions) == 1, "a part after the failure was reported"
        assert verdict.propositions[0].supported is False


class TestTheRepairIsRecordedWhateverHappensToIt:
    """Including when it is refused. A rewrite rejected for inventing a
    number is the validator working, and it was visible only in a log
    line no capture reads."""

    CONTRACT = build_contract(
        "What is SMOTE?",
        QuestionType.DEFINITION,
        entities=("SMOTE",),
    )

    def candidate(self, text: str, quote: str) -> tuple[Any, ...]:
        claim = Claim(
            text=text,
            evidence_ids=["S1-e1"],
            citation_ids=["S1"],
            kind=ClaimKind.FACTUAL,
            answer_slot="definition",
        )
        record = ClaimJudgment(claim_text=text, kind=ClaimKind.FACTUAL, evidence_ids=["S1-e1"])
        verdict = verify_claim(
            text, pairs(quote), ScriptedScorer({}, default=0.0), support_threshold=THRESHOLD
        )
        return (claim, verdict, pairs(quote), "modality", record)

    async def repair(
        self,
        monkeypatch: pytest.MonkeyPatch,
        text: str,
        rewritten: str,
        entailed: dict[str, float],
        quote: str = DEFINITION_QUOTE,
    ) -> ClaimJudgment:
        router = FakeRouter()
        router.responses["RepairOut"] = lambda _user: RepairOut(
            verdicts=[RewriteOut(claim_index=0, rewritten=rewritten)]
        )
        monkeypatch.setattr(reporting, "ctx", lambda: SimpleNamespace(settings=None, router=router))
        candidates = [self.candidate(text, quote)]
        await reporting._repair_wording(
            self.CONTRACT,
            candidates,
            EvidenceStore([], []),
            ScriptedScorer(entailed),
            THRESHOLD,
        )
        return candidates[0][4]

    async def test_an_accepted_repair_keeps_the_wording_it_replaced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reason this record exists. claim.text is reassigned in
        place, so without an explicit copy the published claim's
        previous wording is gone."""
        original = OVERSTATED
        rewritten = WEAKENED
        record = await self.repair(monkeypatch, original, rewritten, {"oversamples": 0.99})

        assert record.repair is not None
        assert record.repair.accepted is True
        assert record.repair.original_text == original
        assert record.repair.repaired_text == rewritten
        assert record.repair.original_text != record.repair.repaired_text

    async def test_a_rewrite_that_invents_a_number_is_recorded_as_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = await self.repair(
            monkeypatch,
            OVERSTATED,
            "SMOTE oversamples the minority class by 40%.",
            {"oversamples": 0.99},
        )

        assert record.repair is not None
        assert record.repair.accepted is False
        assert "number" in record.repair.reason
        # The rejected text is kept: what was proposed is the evidence
        # that the validator had something to refuse.
        assert "40%" in (record.repair.repaired_text or "")

    async def test_a_rewrite_that_is_still_unsupported_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        record = await self.repair(
            monkeypatch,
            OVERSTATED,
            WEAKENED,
            {},  # nothing entails, so the reworded claim fails too
        )
        assert record.repair is not None
        assert record.repair.accepted is False
        assert "still unsupported" in record.repair.reason

    async def test_a_claim_the_model_declined_to_rewrite_is_recorded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Silence is a decision too, and it left no trace at all."""
        router = FakeRouter()
        router.responses["RepairOut"] = lambda _user: RepairOut(verdicts=[])
        monkeypatch.setattr(reporting, "ctx", lambda: SimpleNamespace(settings=None, router=router))
        candidates = [self.candidate(OVERSTATED, DEFINITION_QUOTE)]
        await reporting._repair_wording(
            self.CONTRACT, candidates, EvidenceStore([], []), ScriptedScorer({}), THRESHOLD
        )
        record = candidates[0][4]
        assert record.repair is not None
        assert record.repair.accepted is False
        assert record.repair.reason == "no rewrite was returned"


@pytest.fixture
def faked_io(monkeypatch: pytest.MonkeyPatch) -> FakeRouter:
    """A real graph run over faked I/O and a verifier that entails
    everything, so the run reaches the relevance gate rather than
    stopping at support."""
    import agentic_research.runner as runner

    router = FakeRouter()

    def bundled_report(user: str) -> Any:
        """The default report plus one claim the decomposer splits.

        Built on top of the default rather than in place of it: the
        default's claims are written to clear every gate, so the run
        still reaches the relevance judgement. Replacing them wholesale
        produced a run that withheld everything and a test that passed
        by never getting that far.
        """
        from agentic_research.schemas import ClaimOut
        from fakes import _DEFAULTS

        report = _DEFAULTS["ReportOut"](user)
        offered = re.findall(r"^- (S\d+-e\d+)", user, flags=re.MULTILINE)[:1]
        report.summary_claims.append(
            ClaimOut(
                text=(
                    "SMOTE generates synthetic minority examples, "
                    "and the penalty for missed fraud increases"
                ),
                evidence_ids=offered,
                kind="factual",
                answer_slot="direct_contrast",
            )
        )
        return report

    router.responses["ReportOut"] = bundled_report
    monkeypatch.setattr(
        reporting, "build_verifier", lambda _settings: ScriptedScorer({}, default=0.999)
    )
    monkeypatch.setattr(runner, "ModelRouter", lambda settings, tracker=None: router)

    class _Search(FakeSearchService):
        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

    class _Fetcher(FakeFetcher):
        async def __aenter__(self) -> Any:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

    monkeypatch.setattr(runner, "build_provider", lambda settings: object())
    monkeypatch.setattr(runner, "SearchService", lambda provider, settings: _Search())
    monkeypatch.setattr(runner, "PageFetcher", lambda settings: _Fetcher())
    return router


async def a_run(tmp_path: Any) -> Any:
    return await run_research(
        "How does SMOTE compare with cost-sensitive learning?",
        Settings(
            llm_mode=LLMMode.LOCAL,
            persist_runs=False,
            checkpoint_backend="none",
            output_dir=tmp_path,
            _env_file=None,
        ),
    )


class TestTheCaptureCanReconstructTheRun:
    """End to end, through the serialiser the SSE stream actually uses.

    A field present on the model and absent from the payload is the
    same as no field at all: the hosted capture reads the stream and
    nothing else.
    """

    async def test_the_contract_reaches_the_payload(
        self, faked_io: FakeRouter, tmp_path: Any
    ) -> None:
        payload = serialise_result(await a_run(tmp_path))

        contract = payload["contract"]
        assert contract is not None, "relevance decisions are unreadable without it"
        assert contract["question"]
        assert contract["question_type"]
        assert contract["required_slots"], "a contract with no slots checks nothing"

    async def test_the_relevance_decision_is_structured_not_prose(
        self, faked_io: FakeRouter, tmp_path: Any
    ) -> None:
        payload = serialise_result(await a_run(tmp_path))
        judgments = payload["verification"]["judgments"]

        judged = [j for j in judgments if j.get("relevance")]
        assert judged, "no claim recorded how it passed or failed relevance"
        stages = {j["relevance"]["stage"] for j in judged}
        assert stages <= {"structural", "judged"}
        assert "judged" in stages, "the paid critic call left no record of its verdict"
        for j in judged:
            assert j["relevance"]["relevant"] in (True, False, None)

    async def test_a_decomposed_claim_carries_its_propositions(
        self, faked_io: FakeRouter, tmp_path: Any
    ) -> None:
        payload = serialise_result(await a_run(tmp_path))
        judgments = payload["verification"]["judgments"]

        split = [j for j in judgments if j["propositions"]]
        assert split, "the bundled claim reached the payload with no record of its parts"
        assert any(len(j["propositions"]) > 1 for j in split)
        for j in split:
            for part in j["propositions"]:
                assert part["text"].strip()
                assert 0.0 <= part["best_entailment"] <= 1.0

    async def test_the_payload_still_carries_no_credential(
        self, faked_io: FakeRouter, tmp_path: Any
    ) -> None:
        """The new fields are model-written text and contract slots.
        Neither should be able to carry configuration out."""
        import json

        raw = json.dumps(serialise_result(await a_run(tmp_path)), default=str)
        for marker in ("sk-", "tvly-", "hf_", "redis://"):
            assert marker not in raw, f"{marker} reached the public payload"
        assert not re.search(r"api[_-]?key", raw, flags=re.IGNORECASE)
