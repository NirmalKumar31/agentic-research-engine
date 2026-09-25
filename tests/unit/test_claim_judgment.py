"""The semantic audit record keeps the whole claim.

CitationIssue truncates claim_text at 200 characters, which is right for
a log line and wrong as the only surviving copy of what was judged. Four
rejected claims in the canonical recordings end mid-sentence because the
calibration fixture had to be rebuilt from that field, and the original
wording is unrecoverable -- the run artifact stores the same truncation.

ClaimJudgment exists so that cannot happen again.
"""

from __future__ import annotations

from agentic_research.models import (
    CitationIssue,
    CitationIssueType,
    CitationVerification,
    ClaimJudgment,
    ClaimKind,
)

LONG_CLAIM = (
    "Precision and recall together describe different failure modes under class "
    "imbalance, with precision indicating the proportion of flagged transactions "
    "that are genuinely fraudulent and recall indicating the proportion of actual "
    "fraud cases the model identifies, and neither is adequate alone because a "
    "model can trivially maximise either one at the cost of the other."
)


class TestFullFidelityIsPreserved:
    def test_the_claim_is_long_enough_to_expose_truncation(self) -> None:
        assert len(LONG_CLAIM) > 300

    def test_the_judgment_keeps_every_character(self) -> None:
        record = ClaimJudgment(
            claim_text=LONG_CLAIM,
            kind=ClaimKind.SYNTHESIS,
            evidence_ids=["S1-e1", "S2-e3"],
            verdict="partially_supported",
            reason="scope is wider than the evidence",
            checked=True,
        )
        assert record.claim_text == LONG_CLAIM
        assert len(record.claim_text) == len(LONG_CLAIM)

    def test_the_issue_record_may_still_be_compact(self) -> None:
        """Truncation there is deliberate and stays."""
        issue = CitationIssue(
            type=CitationIssueType.PARTIALLY_SUPPORTED_CLAIM,
            severity="warning",
            claim_text=LONG_CLAIM[:200],
            evidence_id="S1-e1",
            detail="reason",
        )
        assert len(issue.claim_text) == 200
        assert issue.claim_text != LONG_CLAIM

    def test_the_two_records_coexist(self) -> None:
        """One for display, one for audit, and the audit one is complete."""
        verification = CitationVerification(
            issues=[
                CitationIssue(
                    type=CitationIssueType.PARTIALLY_SUPPORTED_CLAIM,
                    severity="warning",
                    claim_text=LONG_CLAIM[:200],
                    evidence_id="S1-e1",
                )
            ],
            judgments=[
                ClaimJudgment(
                    claim_text=LONG_CLAIM,
                    kind=ClaimKind.FACTUAL,
                    evidence_ids=["S1-e1"],
                    verdict="partially_supported",
                    checked=True,
                )
            ],
        )
        assert len(verification.issues[0].claim_text) == 200
        assert verification.judgments[0].claim_text == LONG_CLAIM


class TestUncheckedCandidatesAreRecorded:
    def test_an_unchecked_claim_has_no_verdict(self) -> None:
        """ "Judged not supported" and "never looked at" are different
        findings, even though the gate removes both."""
        record = ClaimJudgment(
            claim_text="A claim nobody reached.",
            kind=ClaimKind.FACTUAL,
            evidence_ids=["S1-e1"],
            reason="not reached within this run's verification budget",
        )
        assert record.verdict is None
        assert record.checked is False

    def test_a_checked_claim_carries_both(self) -> None:
        record = ClaimJudgment(
            claim_text="A claim that was checked.",
            kind=ClaimKind.FACTUAL,
            evidence_ids=["S1-e1"],
            verdict="supported",
            checked=True,
        )
        assert record.verdict == "supported"
        assert record.checked is True


class TestTheEngineRecordsOnePerCandidate:
    def test_every_candidate_path_produces_a_judgment(self) -> None:
        """Uncited, over-long, unreached and checked claims all record
        one, so the audit trail has no silent gaps."""
        import inspect

        from agentic_research.graph.nodes import reporting

        body = inspect.getsource(reporting._check_entailment)
        assert body.count("judgment(") >= 4
