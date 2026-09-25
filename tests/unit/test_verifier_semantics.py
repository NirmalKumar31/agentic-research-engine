"""Semantic failure classes a human claim-by-claim audit found.

Provenance was sound in every case: the quote existed, matched its
source verbatim, and resolved to a real document. What failed was the
*relationship* between quote and claim -- attribution transplanted from
a vendor to a standards body, one study's result generalised to a field,
a recommendation hardened into a requirement.

These tests pin the two mechanisms that make those judgeable: the
verifier is shown who published each quote, and its rules name the
substitutions to reject. They assert the inputs and the rules, not a
model's output -- a model's verdict is not deterministic and a test that
depended on one would be measuring the weather.
"""

from __future__ import annotations

import pytest

from agentic_research.evidence.store import EvidenceStore
from agentic_research.graph.nodes.reporting import _evidence_block
from agentic_research.graph.prompts import VERIFIER_SYSTEM, verifier_user
from agentic_research.models import (
    ContentOrigin,
    EvidenceItem,
    QuoteMatch,
    SourceDocument,
    SourceType,
)


def source(sid: str, title: str, domain: str, kind: SourceType) -> SourceDocument:
    return SourceDocument(
        id=sid,
        url=f"https://{domain}/x",
        canonical_url=f"https://{domain}/x",
        title=title,
        domain=domain,
        text="body",
        content_hash=f"h{sid}",
        source_type=kind,
        content_origin=ContentOrigin.HTML_FETCH,
    )


def evidence(eid: str, sid: str, quote: str, page: int | None = None) -> EvidenceItem:
    return EvidenceItem(
        id=eid,
        source_id=sid,
        sub_question_id="SQ1",
        claim="extracted",
        quote=quote,
        quote_match=QuoteMatch.EXACT_NORMALIZED,
        relevance=0.9,
        page=page,
    )


class TestTheVerifierCanSeeWhoPublishedTheQuote:
    """The NIST case. A claim said the NIST AI RMF *requires* governance
    be assigned to a CISO, CIO or CRO. The quote came from a vendor page
    recommending that structure. Stripped of its publisher, the two read
    identically -- which is how it passed."""

    @pytest.fixture
    def block(self) -> str:
        store = EvidenceStore(
            [source("S1", "AI Governance Best Practices", "vistrada.com", SourceType.VENDOR)],
            [
                evidence(
                    "S1-e2",
                    "S1",
                    "Organisations should assign AI governance ownership to a CISO, CIO or CRO.",
                )
            ],
        )
        return _evidence_block(["S1-e2"], store)

    def test_the_publisher_is_named(self, block: str) -> None:
        assert "vistrada.com" in block

    def test_the_document_kind_is_named(self, block: str) -> None:
        """Labelled "site category", not "type": a retrieval taxonomy is
        not authority, and docs.modulos.ai classifies as official_docs
        without being an official source for NIST."""
        assert "Site category: vendor" in block
        assert "Type: " not in block

    def test_the_title_is_named(self, block: str) -> None:
        assert "AI Governance Best Practices" in block

    def test_the_quote_still_appears_verbatim(self, block: str) -> None:
        assert "assign AI governance ownership to a CISO, CIO or CRO" in block

    def test_a_standards_source_is_distinguishable_from_a_vendor(self) -> None:
        """The distinction the claim turned on has to be visible."""
        store = EvidenceStore(
            [
                source("S1", "Vendor guidance", "vistrada.com", SourceType.VENDOR),
                source("S4", "NIST AI RMF 1.0", "nvlpubs.nist.gov", SourceType.STANDARDS_BODY),
            ],
            [
                evidence("S1-e1", "S1", "Assign governance to a CISO."),
                evidence("S4-e1", "S4", "Govern is one of four core functions.", page=21),
            ],
        )
        block = _evidence_block(["S1-e1", "S4-e1"], store)
        assert "Site category: vendor" in block
        assert "Site category: standards_body" in block
        assert "Page: 21" in block

    def test_quality_score_is_not_offered_as_confidence(self, block: str) -> None:
        """A retrieval heuristic about document type and rank, which a
        verifier would read as confidence in the claim."""
        assert "quality" not in block.lower()
        assert "0.5" not in block and "0.6" not in block


class TestTheRulesNameTheSubstitutionsToReject:
    """Each rule corresponds to a claim the audit caught."""

    @pytest.mark.parametrize(
        ("phrase", "case"),
        [
            ("different publisher", "NIST requirement sourced from a vendor page"),
            ("general statement about the field", "one paper's result stated as universal"),
            ("'does', 'will' or 'always'", "may/can hardened into does/will"),
            ("guidance or a recommendation into a requirement", "advice stated as a mandate"),
            ("causation", "association reported as cause"),
            ("threshold, ranking, superlative or quantity", "'100+ documents', 'most effective'"),
            ("any material part is unsupported", "compound claim, one clause unevidenced"),
        ],
    )
    def test_each_failure_class_is_covered(self, phrase: str, case: str) -> None:
        assert phrase in VERIFIER_SYSTEM, f"no rule covers: {case}"

    def test_a_vendor_is_not_the_standard_it_describes(self) -> None:
        assert "a vendor describing what a standard requires is not the standard" in (
            VERIFIER_SYSTEM
        )

    def test_outside_knowledge_is_still_forbidden(self) -> None:
        """Strengthening the rules must not turn the verifier into a
        fact-checker; it judges the quote in front of it."""
        assert "Do not use outside" in VERIFIER_SYSTEM

    def test_the_question_asks_about_these_sources(self) -> None:
        prompt = verifier_user("a claim", "a block")
        assert "from these sources" in prompt


class TestTheAuditedClaimShapes:
    """The four shapes the audit flagged, as data rather than prose, so
    the evidence a verifier sees for each is exercised end to end."""

    def test_attributed_requirement_from_a_third_party(self) -> None:
        store = EvidenceStore(
            [source("S2", "Vistrada blog", "vistrada.com", SourceType.VENDOR)],
            [evidence("S2-e1", "S2", "We recommend a vCISO own AI governance.")],
        )
        block = _evidence_block(["S2-e1"], store)
        # The verifier can see the quote is a recommendation by a vendor,
        # not a requirement stated by NIST.
        assert "Domain: vistrada.com" in block
        assert "recommend" in block

    def test_compound_claim_shows_every_cited_quote(self) -> None:
        """The compound NIST claim joined a vCISO point to inventory
        complexity and AI inaccuracy; only the first was evidenced. The
        verifier must see all cited evidence to notice the gap."""
        store = EvidenceStore(
            [source("S2", "Vistrada blog", "vistrada.com", SourceType.VENDOR)],
            [
                evidence("S2-e1", "S2", "A vCISO can cover the gap."),
                evidence("S2-e2", "S2", "Asset inventories are hard to maintain."),
            ],
        )
        block = _evidence_block(["S2-e1", "S2-e2"], store)
        assert "S2-e1" in block and "S2-e2" in block

    def test_scope_claim_shows_the_study_it_came_from(self) -> None:
        """'Ensemble methods are the most effective' vs a paper reporting
        that ensembles performed best in its own experiment."""
        store = EvidenceStore(
            [source("S3", "A comparative study", "mdpi.com", SourceType.ACADEMIC)],
            [
                evidence(
                    "S3-e1",
                    "S3",
                    "In our experiments, ensemble methods achieved the highest F1 score.",
                )
            ],
        )
        block = _evidence_block(["S3-e1"], store)
        assert "In our experiments" in block
        assert "Site category: academic" in block

    def test_unevidenced_consequence_is_visible_as_absent(self) -> None:
        """Evidence establishes that fraud patterns evolve and labels get
        harder; the claim asserted significant retraining. The verifier
        sees only what was cited."""
        store = EvidenceStore(
            [source("S5", "Fraud detection survey", "sciencedirect.com", SourceType.ACADEMIC)],
            [
                evidence(
                    "S5-e1",
                    "S5",
                    "Fraud patterns evolve and labelled data becomes harder to obtain.",
                )
            ],
        )
        block = _evidence_block(["S5-e1"], store)
        assert "retrain" not in block.lower()
