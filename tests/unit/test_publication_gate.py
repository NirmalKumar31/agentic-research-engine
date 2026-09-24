"""The publication gate: only supported claims are published.

Two defects motivated this, both seen in real output.

A recording published:

    "Vector databases become most favorable for RAG applications with
     minimum data requirements of 100+ documents..."

cited to evidence saying only that RAG is commonly used for internal
knowledge bots. The threshold is not in the source.

Then a bounded live run checked 1 of 6 eligible claims and published all
six. The gate removed claims carrying a *failing* verdict, so a claim the
verifier never reached carried no issue and survived. A deny-list cannot
express "not verified"; the gate is an allow-list for that reason.

It removes; it does not rewrite. Nothing is re-asked of a model and no
replacement prose is invented.
"""

from __future__ import annotations

import pytest

from agentic_research.citations.publication import (
    ClaimKey,
    ClaimVerdict,
    claim_key,
    filter_report_by_verification,
    key_of,
)
from agentic_research.models import (
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


def claim(text: str, evidence_ids: list[str], kind: ClaimKind = ClaimKind.FACTUAL) -> Claim:
    return Claim(text=text, evidence_ids=evidence_ids, citation_ids=[], kind=kind)


def verdicts_for(*pairs: tuple[Claim, ClaimVerdict]) -> dict[ClaimKey, ClaimVerdict]:
    return {key_of(c): v for c, v in pairs}


def published(report: ResearchReport) -> list[str]:
    out = [c.text for c in report.summary_claims] + [c.text for c in report.key_findings]
    for section in report.sections:
        out += [c.text for c in section.claims]
    return out


class TestOnlySupportedClaimsSurvive:
    def test_a_supported_claim_is_published(self) -> None:
        good = claim("Precision-recall suits heavy imbalance.", ["S1-e1"])
        report = ResearchReport(title="T", summary_claims=[good])

        filtered, removed = filter_report_by_verification(report, verdicts_for((good, "supported")))
        assert removed == 0
        assert published(filtered) == [good.text]

    @pytest.mark.parametrize("verdict", ["partially_supported", "unsupported"])
    def test_a_failing_verdict_removes_the_claim(self, verdict: ClaimVerdict) -> None:
        bad = claim(BAD_CLAIM, [BAD_EVIDENCE_ID])
        report = ResearchReport(title="T", summary_claims=[bad])

        filtered, removed = filter_report_by_verification(report, verdicts_for((bad, verdict)))
        assert removed == 1
        assert published(filtered) == []

    def test_an_unchecked_claim_is_not_published(self) -> None:
        """The live defect. No verdict is not the same as no objection."""
        never_checked = claim("Something the verifier never reached.", ["S1-e1"])
        report = ResearchReport(title="T", summary_claims=[never_checked])

        filtered, removed = filter_report_by_verification(report, {})
        assert removed == 1
        assert published(filtered) == []

    def test_a_substantive_claim_with_no_evidence_is_not_published(self) -> None:
        """It can never be checked, so it can never earn publication."""
        uncited = claim("An assertion with nothing behind it.", [])
        report = ResearchReport(title="T", summary_claims=[uncited])

        filtered, removed = filter_report_by_verification(report, {})
        assert removed == 1
        assert published(filtered) == []

    def test_framing_survives_without_a_verdict(self) -> None:
        framing = claim("This report compares two approaches.", [], ClaimKind.FRAMING)
        report = ResearchReport(title="T", summary_claims=[framing])

        filtered, removed = filter_report_by_verification(report, {})
        assert removed == 0
        assert published(filtered) == [framing.text]


class TestTheBoundedLiveScenario:
    """12 checkable, 10 selected, 8 supported, 1 partial, 1 unsupported,
    2 never reached. Exactly 8 substantive claims may be published."""

    @staticmethod
    def _build() -> tuple[ResearchReport, dict[ClaimKey, ClaimVerdict]]:
        supported = [claim(f"Supported finding {i}.", [f"S1-e{i}"]) for i in range(8)]
        partial = claim("Partially supported finding.", ["S1-e8"])
        unsupported = claim("Unsupported finding.", ["S1-e9"])
        unchecked = [claim(f"Unreached finding {i}.", [f"S1-e{10 + i}"]) for i in range(2)]

        report = ResearchReport(
            title="T",
            summary_claims=supported[:3],
            key_findings=supported[3:6],
            sections=[
                ReportSection(heading="A", claims=[*supported[6:], partial]),
                ReportSection(heading="B", claims=[unsupported, *unchecked]),
            ],
        )
        verdicts = verdicts_for(
            *[(c, "supported") for c in supported],
            (partial, "partially_supported"),
            (unsupported, "unsupported"),
        )
        return report, verdicts

    def test_exactly_the_eight_supported_claims_are_published(self) -> None:
        report, verdicts = self._build()
        assert len(report.substantive_claims()) == 12

        filtered, removed = filter_report_by_verification(report, verdicts)

        assert len(filtered.substantive_claims()) == 8
        assert removed == 4, "1 partial + 1 unsupported + 2 unchecked"
        assert all(t.startswith("Supported finding") for t in published(filtered))

    def test_the_two_unchecked_claims_do_not_survive(self) -> None:
        report, verdicts = self._build()
        filtered, _ = filter_report_by_verification(report, verdicts)
        assert not any("Unreached" in t for t in published(filtered))

    def test_a_section_emptied_by_the_gate_is_dropped(self) -> None:
        report, verdicts = self._build()
        filtered, _ = filter_report_by_verification(report, verdicts)
        # Section B held only an unsupported claim and two unchecked ones.
        assert [s.heading for s in filtered.sections] == ["A"]

    def test_the_removal_is_disclosed_in_the_limitations(self) -> None:
        report, verdicts = self._build()
        filtered, removed = filter_report_by_verification(report, verdicts)
        assert any(str(removed) in limit for limit in filtered.limitations)


class TestIdentityIsExactNotFuzzy:
    def test_the_same_text_with_different_evidence_is_a_different_claim(self) -> None:
        a = claim("Identical wording.", ["S1-e1"])
        b = claim("Identical wording.", ["S2-e9"])
        report = ResearchReport(title="T", summary_claims=[a], key_findings=[b])

        filtered, removed = filter_report_by_verification(
            report, verdicts_for((a, "supported"), (b, "unsupported"))
        )
        assert removed == 1
        assert [c.evidence_ids for c in filtered.summary_claims] == [["S1-e1"]]
        assert filtered.key_findings == []

    def test_a_claim_repeated_verbatim_is_resolved_once(self) -> None:
        """One key, so both copies share a verdict and move together."""
        repeated = claim("Stated in two places.", ["S1-e1"])
        report = ResearchReport(
            title="T",
            summary_claims=[repeated],
            sections=[ReportSection(heading="A", claims=[repeated])],
        )
        filtered, removed = filter_report_by_verification(
            report, verdicts_for((repeated, "unsupported"))
        )
        assert removed == 2
        assert published(filtered) == []

    def test_the_key_truncates_where_the_issue_record_truncates(self) -> None:
        long_text = "x" * 400
        assert claim_key(long_text, ["S1-e1"]) == (long_text[:200], "S1-e1")


class TestTheMotivatingRegression:
    def test_the_100_documents_claim_cannot_be_published_unverified(self) -> None:
        bad = claim(BAD_CLAIM, [BAD_EVIDENCE_ID])
        report = ResearchReport(title="T", summary_claims=[bad])

        # Not checked at all: previously this published.
        filtered, _ = filter_report_by_verification(report, {})
        assert BAD_CLAIM not in published(filtered)


class TestNothingIsRewritten:
    def test_surviving_claim_text_is_untouched(self) -> None:
        good = claim("Exact wording preserved.", ["S1-e1"])
        report = ResearchReport(title="T", summary_claims=[good])
        filtered, _ = filter_report_by_verification(report, verdicts_for((good, "supported")))
        assert filtered.summary_claims[0].text == "Exact wording preserved."
        assert filtered.summary_claims[0].evidence_ids == ["S1-e1"]

    def test_an_unfiltered_report_is_returned_unchanged(self) -> None:
        good = claim("All fine.", ["S1-e1"])
        report = ResearchReport(title="T", summary_claims=[good], limitations=["pre-existing"])
        filtered, removed = filter_report_by_verification(report, verdicts_for((good, "supported")))
        assert removed == 0
        assert filtered is report
        assert filtered.limitations == ["pre-existing"]
