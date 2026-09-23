"""PDF extraction, page provenance and page-level citations.

PDFs were previously skipped as unsupported content, which quietly excluded
much of the academic literature from every run.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from agentic_research.config import Settings
from agentic_research.evidence.store import classify_quote
from agentic_research.models import (
    Claim,
    ContentOrigin,
    DiscoveryRef,
    EvidenceItem,
    FetchStatus,
    QuoteMatch,
    ResearchReport,
    SourceDocument,
)
from agentic_research.report import render_markdown
from agentic_research.retrieval import PageFetcher
from agentic_research.retrieval.pdf import (
    PdfExtractionError,
    ScannedPdfError,
    extract_pdf,
    looks_like_pdf,
)
from pdf_fixtures import make_malformed_pdf, make_oversized_pdf, make_pdf, make_scanned_pdf

PAGES = [
    "Class imbalance dominates fraud detection datasets in practice today.",
    "Precision recall curves are preferred over ROC AUC under heavy imbalance.",
    "Cost sensitive learning penalises missed fraud more heavily than false alarms.",
    "Gradient boosted trees remain a strong production baseline for this task.",
]


@pytest.fixture(autouse=True)
def _pin_dns(stub_dns: str) -> None:
    """The fetcher pins validated addresses, so DNS must be deterministic."""


class TestDetection:
    def test_magic_bytes_win_over_a_missing_suffix(self) -> None:
        """Plenty of PDFs are served from URLs with no .pdf on the end."""
        assert looks_like_pdf(make_pdf(["x"]), "application/octet-stream")

    def test_content_type_alone_is_enough(self) -> None:
        assert looks_like_pdf(b"no magic here", "application/pdf")

    def test_an_html_error_page_at_a_pdf_url_is_not_a_pdf(self) -> None:
        """A .pdf URL routinely returns an HTML 404 body with a 200 status."""
        assert not looks_like_pdf(b"<html><body>Not found</body></html>", "text/html")

    def test_leading_bytes_before_the_header_are_tolerated(self) -> None:
        assert looks_like_pdf(b"\xef\xbb\xbf\n" + make_pdf(["x"]))


class TestExtraction:
    def test_single_page(self) -> None:
        document = extract_pdf(make_pdf([PAGES[0]]))
        assert "Class imbalance dominates" in document.text
        assert document.page_count == 1
        assert document.pages_extracted == 1

    def test_multi_page_text_is_joined_in_order(self) -> None:
        document = extract_pdf(make_pdf(PAGES))
        assert document.page_count == 4
        positions = [document.text.find(p.split()[0]) for p in PAGES]
        assert positions == sorted(positions), "pages must stay in document order"

    def test_malformed_pdf_raises_rather_than_returning_garbage(self) -> None:
        with pytest.raises(PdfExtractionError):
            extract_pdf(make_malformed_pdf())

    def test_scanned_pdf_is_distinguished_from_a_broken_one(self) -> None:
        """The remedy differs: one needs OCR, the other is simply unusable."""
        with pytest.raises(ScannedPdfError):
            extract_pdf(make_scanned_pdf())

    def test_page_cap_is_enforced(self) -> None:
        document = extract_pdf(make_pdf(PAGES * 10), max_pages=5)
        assert document.page_count == 40
        assert document.pages_extracted <= 5
        assert document.truncated

    def test_character_cap_is_enforced(self) -> None:
        document = extract_pdf(make_pdf(PAGES * 10), max_chars=200)
        assert len(document.text) <= 220
        assert document.truncated


class TestPageProvenance:
    def test_offset_maps_back_to_the_right_page(self) -> None:
        document = extract_pdf(make_pdf(PAGES))
        for expected_page, page_text in enumerate(PAGES, start=1):
            offset = document.text.find(page_text.split()[0])
            assert document.page_of_offset(offset) == expected_page

    def test_quote_location_yields_a_page_number(self) -> None:
        """The chain a PDF citation depends on: quote -> offset -> page."""
        document = extract_pdf(make_pdf(PAGES))
        match, offset = classify_quote("preferred over ROC AUC under heavy", document.text)
        assert match is QuoteMatch.EXACT_NORMALIZED
        assert offset is not None
        assert document.page_of_offset(offset) == 2

    def test_source_document_maps_offsets_too(self) -> None:
        document = extract_pdf(make_pdf(PAGES))
        source = SourceDocument(
            id="S7",
            url="https://x.org/p.pdf",
            canonical_url="https://x.org/p.pdf",
            title="Paper",
            domain="x.org",
            text=document.text,
            page_offsets=document.page_offsets,
            content_origin=ContentOrigin.PDF_EXTRACT,
        )
        offset = document.text.find("Gradient boosted")
        assert source.page_of_offset(offset) == 4


class TestFetcherIntegration:
    @respx.mock
    async def test_a_pdf_is_fetched_and_extracted(self, settings: Settings) -> None:
        respx.get(path="/paper.pdf").mock(
            return_value=httpx.Response(
                200, content=make_pdf(PAGES), headers={"content-type": "application/pdf"}
            )
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://x.org/paper.pdf")

        assert result.ok
        assert result.content_origin is ContentOrigin.PDF_EXTRACT
        assert len(result.page_offsets) == 4
        assert "Precision recall" in result.text

    @respx.mock
    async def test_a_pdf_without_a_suffix_is_still_extracted(self, settings: Settings) -> None:
        respx.get(path="/download").mock(
            return_value=httpx.Response(
                200,
                content=make_pdf(PAGES),
                headers={"content-type": "application/octet-stream"},
            )
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://x.org/download?id=9")
        assert result.ok
        assert result.content_origin is ContentOrigin.PDF_EXTRACT

    @respx.mock
    async def test_scanned_pdf_is_classified_and_the_run_continues(
        self, settings: Settings
    ) -> None:
        respx.get(path="/scan.pdf").mock(
            return_value=httpx.Response(
                200, content=make_scanned_pdf(), headers={"content-type": "application/pdf"}
            )
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://x.org/scan.pdf")
        assert result.status is FetchStatus.SCANNED_PDF
        assert not result.ok
        assert "OCR" in (result.error or "")

    @respx.mock
    async def test_malformed_pdf_fails_safely(self, settings: Settings) -> None:
        respx.get(path="/bad.pdf").mock(
            return_value=httpx.Response(
                200, content=make_malformed_pdf(), headers={"content-type": "application/pdf"}
            )
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://x.org/bad.pdf")
        assert result.status is FetchStatus.ERROR
        assert result.text == ""

    @respx.mock
    async def test_oversized_pdf_is_rejected_by_the_pdf_cap(self, settings: Settings) -> None:
        """PDFs get a larger cap than HTML, but still a cap."""
        settings.max_pdf_bytes = 50_000
        respx.get(path="/huge.pdf").mock(
            return_value=httpx.Response(
                200,
                content=make_oversized_pdf(200_000),
                headers={"content-type": "application/pdf"},
            )
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://x.org/huge.pdf")
        assert result.status is FetchStatus.TOO_LARGE

    @respx.mock
    async def test_pdfs_get_a_larger_cap_than_html(self, settings: Settings) -> None:
        settings.max_page_bytes = 20_000
        settings.max_pdf_bytes = 500_000
        respx.get(path="/mid.pdf").mock(
            return_value=httpx.Response(
                200,
                content=make_oversized_pdf(60_000),
                headers={"content-type": "application/pdf"},
            )
        )
        async with PageFetcher(settings) as fetcher:
            result = await fetcher.fetch("https://x.org/mid.pdf")
        assert result.ok, "a 60KB PDF must pass under a 500KB PDF cap"


class TestPageCitationRendering:
    def test_a_pdf_citation_renders_its_page(self) -> None:
        source = SourceDocument(
            id="S7",
            url="https://x.org/p.pdf",
            canonical_url="https://x.org/p.pdf",
            title="Paper",
            domain="x.org",
            text="body",
            content_hash="h",
            content_origin=ContentOrigin.PDF_EXTRACT,
        )
        item = EvidenceItem(
            id="S7-e1",
            source_id="S7",
            sub_question_id="SQ1",
            claim="c",
            quote="q",
            quote_match=QuoteMatch.EXACT_NORMALIZED,
            relevance=0.9,
            page=14,
            discovery=DiscoveryRef(query_id="Q1", sub_question_id="SQ1"),
        )
        report = ResearchReport(
            title="T",
            key_findings=[
                Claim(text="Imbalance dominates", evidence_ids=["S7-e1"], citation_ids=["S7"])
            ],
        )
        markdown = render_markdown(report, [source], None, evidence=[item])
        assert "[S7, p. 14]" in markdown

    def test_a_web_citation_has_no_page_number(self) -> None:
        """Page numbers are only rendered where genuinely known."""
        source = SourceDocument(
            id="S1",
            url="https://x.org/a",
            canonical_url="https://x.org/a",
            title="Page",
            domain="x.org",
            text="body",
            content_hash="h",
            content_origin=ContentOrigin.HTML_FETCH,
        )
        item = EvidenceItem(
            id="S1-e1",
            source_id="S1",
            sub_question_id="SQ1",
            claim="c",
            quote="q",
            quote_match=QuoteMatch.EXACT_NORMALIZED,
            relevance=0.9,
        )
        report = ResearchReport(
            title="T",
            key_findings=[Claim(text="Something", evidence_ids=["S1-e1"], citation_ids=["S1"])],
        )
        markdown = render_markdown(report, [source], None, evidence=[item])
        assert "[S1]" in markdown
        assert "p. " not in markdown
