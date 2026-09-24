"""The publication gate: claims the evidence did not support are removed.

A real recording published this:

    "Vector databases become most favorable for RAG applications with
     minimum data requirements of 100+ documents..."

cited to evidence that says only that RAG is commonly used for internal
knowledge bots. The threshold is not in the source. A provenance demo that
publishes a number its own evidence does not contain undercuts the entire
claim of the project, so the gate exists to make that impossible.

It removes; it does not rewrite. Nothing is re-asked of a model and no
replacement prose is invented.
"""

from __future__ import annotations

import pytest

from agentic_research.citations.publication import (
    filter_report_by_verification,
    rejected_keys,
)
from agentic_research.models import (
    CitationIssue,
    CitationIssueType,
    CitationVerification,
    Claim,
    ClaimKind,
    ReportSection,
    ResearchReport,
)

# The claim and evidence that motivated this gate, kept verbatim.
BAD_CLAIM = (
    "Vector databases become most favorable for RAG applications with minimum "
    "data requirements of 100+ documents and sub-second latency needs."
)
BAD_EVIDENCE_ID = "S2-e5"


def _issue(kind: CitationIssueType, text: str, evidence_ids: list[str]) -> CitationIssue:
    """Built the way the verifier builds it, including the truncation."""
    return CitationIssue(
        type=kind,
        severity="warning",
        claim_text=text[:200],
        evidence_id=",".join(evidence_ids),
        detail="the cited evidence does not establish the threshold",
    )


class TestTheKnownRegression:
    def test_the_hundred_documents_claim_is_removed(self) -> None:
        bad = Claim(text=BAD_CLAIM, evidence_ids=[BAD_EVIDENCE_ID], citation_ids=["S2"])
        good = Claim(
            text="Vector databases are commonly used to power internal knowledge bots.",
            evidence_ids=["S2-e1"],
            citation_ids=["S2"],
        )
        report = ResearchReport(title="T", summary_claims=[bad], key_findings=[good])
        verification = CitationVerification(
            issues=[_issue(CitationIssueType.UNSUPPORTED_CLAIM, BAD_CLAIM, [BAD_EVIDENCE_ID])]
        )

        filtered, removed = filter_report_by_verification(report, verification)

        assert removed == 1
        texts = [c.text for c in filtered.all_claims()]
        assert BAD_CLAIM not in texts
        assert good.text in texts

    def test_the_removal_does_not_erase_the_audit_trail(self) -> None:
        """A technical reviewer must still be able to see what failed and
        why, after the claim has gone from the report."""
        report = ResearchReport(
            title="T",
            summary_claims=[
                Claim(text=BAD_CLAIM, evidence_ids=[BAD_EVIDENCE_ID], citation_ids=["S2"])
            ],
        )
        verification = CitationVerification(
            issues=[_issue(CitationIssueType.UNSUPPORTED_CLAIM, BAD_CLAIM, [BAD_EVIDENCE_ID])]
        )

        filtered, _ = filter_report_by_verification(report, verification)

        assert filtered.all_claims() == []
        # The issue survives untouched.
        assert len(verification.issues) == 1
        assert verification.issues[0].type is CitationIssueType.UNSUPPORTED_CLAIM
        assert BAD_CLAIM.startswith(verification.issues[0].claim_text[:60])

    def test_it_never_appears_in_the_rendered_markdown(self) -> None:
        from agentic_research.report import render_markdown

        report = ResearchReport(
            title="T",
            summary_claims=[
                Claim(text=BAD_CLAIM, evidence_ids=[BAD_EVIDENCE_ID], citation_ids=["S2"])
            ],
        )
        verification = CitationVerification(
            issues=[_issue(CitationIssueType.UNSUPPORTED_CLAIM, BAD_CLAIM, [BAD_EVIDENCE_ID])]
        )
        filtered, _ = filter_report_by_verification(report, verification)
        markdown = render_markdown(filtered, [], None, evidence=[])
        assert "100+ documents" not in markdown


class TestVerdictHandling:
    def test_a_supported_claim_is_left_exactly_as_written(self) -> None:
        claim = Claim(text="A supported claim.", evidence_ids=["S1-e1"], citation_ids=["S1"])
        report = ResearchReport(title="T", summary_claims=[claim])

        filtered, removed = filter_report_by_verification(report, CitationVerification())

        assert removed == 0
        assert filtered.summary_claims[0].text == "A supported claim."
        assert filtered.limitations == []

    def test_a_partially_supported_claim_is_excluded_not_softened(self) -> None:
        """Conservative on purpose: "mostly true" reads to a reader exactly
        like "true", and the report offers no way to tell them apart."""
        compound = "Latency improves and costs fall by half."
        report = ResearchReport(
            title="T",
            summary_claims=[Claim(text=compound, evidence_ids=["S1-e1"], citation_ids=["S1"])],
        )
        verification = CitationVerification(
            issues=[_issue(CitationIssueType.PARTIALLY_SUPPORTED_CLAIM, compound, ["S1-e1"])]
        )

        filtered, removed = filter_report_by_verification(report, verification)

        assert removed == 1
        assert filtered.all_claims() == []
        # Not rewritten into a narrower version of itself.
        assert all(compound[:30] not in c.text for c in filtered.all_claims())

    def test_framing_claims_are_never_gated_on_evidence(self) -> None:
        """They carry none by design; removing them would be a bug."""
        framing = Claim(text="This section compares the two.", kind=ClaimKind.FRAMING)
        report = ResearchReport(title="T", sections=[ReportSection(heading="H", claims=[framing])])

        filtered, removed = filter_report_by_verification(report, CitationVerification())

        assert removed == 0
        assert filtered.sections[0].claims[0].text == framing.text

    def test_other_issue_types_do_not_remove_claims(self) -> None:
        """An unused source or a redundant citation is not grounds for
        deleting a sentence."""
        claim = Claim(text="A claim.", evidence_ids=["S1-e1"], citation_ids=["S1"])
        report = ResearchReport(title="T", summary_claims=[claim])
        verification = CitationVerification(
            issues=[_issue(CitationIssueType.UNUSED_SOURCE, "A claim.", ["S1-e1"])]
        )

        _, removed = filter_report_by_verification(report, verification)
        assert removed == 0


class TestFilteringIsConsistentEverywhere:
    def test_a_restated_claim_is_removed_from_every_location(self) -> None:
        """Leaving one copy would publish the exact text that failed."""
        claim = Claim(text=BAD_CLAIM, evidence_ids=[BAD_EVIDENCE_ID], citation_ids=["S2"])
        report = ResearchReport(
            title="T",
            summary_claims=[claim],
            key_findings=[claim],
            sections=[ReportSection(heading="Detail", claims=[claim])],
        )
        verification = CitationVerification(
            issues=[_issue(CitationIssueType.UNSUPPORTED_CLAIM, BAD_CLAIM, [BAD_EVIDENCE_ID])]
        )

        filtered, removed = filter_report_by_verification(report, verification)

        assert removed == 3
        assert filtered.all_claims() == []

    def test_a_section_left_empty_is_dropped(self) -> None:
        report = ResearchReport(
            title="T",
            sections=[
                ReportSection(
                    heading="Gone",
                    claims=[Claim(text=BAD_CLAIM, evidence_ids=["S2-e5"], citation_ids=["S2"])],
                ),
                ReportSection(
                    heading="Kept",
                    claims=[Claim(text="Fine.", evidence_ids=["S1-e1"], citation_ids=["S1"])],
                ),
            ],
        )
        verification = CitationVerification(
            issues=[_issue(CitationIssueType.UNSUPPORTED_CLAIM, BAD_CLAIM, ["S2-e5"])]
        )

        filtered, _ = filter_report_by_verification(report, verification)

        assert [s.heading for s in filtered.sections] == ["Kept"]

    def test_removal_adds_one_plain_limitation(self) -> None:
        """No internal labels, no verifier reasoning, no SQ tokens."""
        report = ResearchReport(
            title="T",
            summary_claims=[
                Claim(text=BAD_CLAIM, evidence_ids=["S2-e5"], citation_ids=["S2"]),
                Claim(text="Another bad one.", evidence_ids=["S3-e1"], citation_ids=["S3"]),
            ],
        )
        verification = CitationVerification(
            issues=[
                _issue(CitationIssueType.UNSUPPORTED_CLAIM, BAD_CLAIM, ["S2-e5"]),
                _issue(CitationIssueType.UNSUPPORTED_CLAIM, "Another bad one.", ["S3-e1"]),
            ]
        )

        filtered, removed = filter_report_by_verification(report, verification)

        assert removed == 2
        note = filtered.limitations[-1]
        assert note == (
            "2 generated claim(s) were excluded because the cited evidence "
            "did not fully support them."
        )
        for forbidden in ("S2-e5", "SQ", "unsupported", "verifier"):
            assert forbidden not in note

    def test_identity_uses_evidence_not_text_alone(self) -> None:
        """Two claims can share wording while citing different evidence;
        only the one that failed should go."""
        text = "Latency improves under load."
        failed = Claim(text=text, evidence_ids=["S1-e1"], citation_ids=["S1"])
        other = Claim(text=text, evidence_ids=["S9-e9"], citation_ids=["S9"])
        report = ResearchReport(title="T", summary_claims=[failed], key_findings=[other])
        verification = CitationVerification(
            issues=[_issue(CitationIssueType.UNSUPPORTED_CLAIM, text, ["S1-e1"])]
        )

        filtered, removed = filter_report_by_verification(report, verification)

        assert removed == 1
        assert filtered.key_findings[0].evidence_ids == ["S9-e9"]
        assert filtered.summary_claims == []


class TestRejectedKeys:
    def test_collects_both_failing_verdicts(self) -> None:
        verification = CitationVerification(
            issues=[
                _issue(CitationIssueType.UNSUPPORTED_CLAIM, "a", ["S1-e1"]),
                _issue(CitationIssueType.PARTIALLY_SUPPORTED_CLAIM, "b", ["S2-e1"]),
                _issue(CitationIssueType.UNUSED_SOURCE, "c", ["S3-e1"]),
            ]
        )
        keys = rejected_keys(verification)
        assert ("a", "S1-e1") in keys
        assert ("b", "S2-e1") in keys
        assert ("c", "S3-e1") not in keys


class TestPublicReportsCarryNoInternalTokens:
    """Coverage gaps are identifiers internally and must not stay that way.

    A real recording published this limitation verbatim:

        "no evidence for For SQ1: The evidence mentions performance metrics
         but doesn't provide specific numbers..."

    Two faults in one line. "no evidence for SQ1" means nothing to a
    reader, and the rest is the critic's own reasoning, returned in a field
    meant to hold sub-question identifiers.
    """

    @staticmethod
    def _limitations(missing: list[str], weak: list[str], questions: list[tuple[str, str]]):
        from agentic_research.graph.nodes.reporting import _coverage_limitations
        from agentic_research.models import CoverageAssessment, SubQuestion

        coverage = CoverageAssessment(round_number=1, missing=missing, weak=weak)
        subs = [SubQuestion(id=i, text=t, rationale="r") for i, t in questions]
        return _coverage_limitations(coverage, subs)

    def test_a_gap_is_named_by_its_question_not_its_identifier(self) -> None:
        out = self._limitations(["SQ3"], [], [("SQ3", "How do the security implications compare?")])
        assert out == [
            "The retrieved evidence did not answer: How do the security implications compare."
        ]
        assert not any("SQ3" in line for line in out)

    def test_a_thin_dimension_reads_as_a_sentence(self) -> None:
        out = self._limitations([], ["SQ1"], [("SQ1", "What are the latency characteristics?")])
        assert out == ["Only limited evidence was found for: What are the latency characteristics."]

    def test_critic_reasoning_in_the_gap_field_is_dropped(self) -> None:
        """The field holds identifiers. Prose in it is the model thinking
        out loud, and publishing that is worse than publishing nothing."""
        reasoning = (
            "For SQ1: The evidence mentions performance metrics but doesn't provide "
            "specific numbers, and I need to check if there are critical gaps."
        )
        out = self._limitations([reasoning], [], [("SQ1", "What are the latency traits?")])
        assert out == []

    @pytest.mark.parametrize(
        "forbidden",
        ["no evidence for SQ", "thin evidence for SQ", "I need to check", "I should verify"],
    )
    def test_forbidden_phrases_never_reach_a_rendered_report(self, forbidden: str) -> None:
        from agentic_research.report import render_markdown

        report = ResearchReport(
            title="T",
            summary_claims=[Claim(text="A claim.", evidence_ids=["S1-e1"], citation_ids=["S1"])],
            limitations=self._limitations(
                [f"{forbidden} something"], [], [("SQ1", "A real question?")]
            ),
        )
        markdown = render_markdown(report, [], None, evidence=[])
        assert forbidden not in markdown

    def test_the_committed_recordings_contain_no_internal_tokens(self) -> None:
        """Runs against the shipped recordings, so a future one cannot
        reintroduce this silently."""
        import json
        import re

        from agentic_research.web.recordings import RECORDINGS_DIR

        patterns = [
            re.compile(r"no evidence for SQ\d+"),
            re.compile(r"thin evidence for SQ\d+"),
            re.compile(r"\bI need to check\b"),
            re.compile(r"\bI should verify\b"),
        ]
        offenders: list[str] = []
        for path in sorted(RECORDINGS_DIR.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            report = payload.get("result", {}).get("report") or {}
            markdown = payload.get("result", {}).get("markdown", "")
            text = json.dumps(report) + markdown
            for pattern in patterns:
                if pattern.search(text):
                    offenders.append(f"{path.name}: {pattern.pattern}")
        assert offenders == [], offenders
