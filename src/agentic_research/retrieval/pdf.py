"""PDF text extraction with page provenance.

A lot of the best research sources are PDFs, and the engine previously
skipped them as unsupported content. Extraction here is page-aware: the
offset at which each page's text begins is recorded, so a quote located in
the assembled text can be mapped back to the page it came from and a
citation can render ``[S7, p. 14]`` without guessing.

Uses ``pypdf`` (BSD-3-Clause), which is pure Python, has no system
dependencies, and does text extraction only. No OCR: a scanned PDF is
detected and reported as such rather than silently yielding nothing.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError

from agentic_research.observability import get_logger
from agentic_research.retrieval.parser import clean_text

log = get_logger(__name__)

PDF_MAGIC = b"%PDF-"

# Below this many characters per page on average, the document is almost
# certainly scanned images rather than text.
_MIN_CHARS_PER_PAGE = 40


class PdfExtractionError(Exception):
    """The PDF could not be read at all."""


class ScannedPdfError(PdfExtractionError):
    """The PDF parsed but holds essentially no extractable text.

    Distinguished from a malformed file because the remedy differs: this one
    needs OCR, which is deliberately out of scope, and the run should
    continue with the source marked unusable rather than treated as an error.
    """


@dataclass
class PdfDocument:
    text: str
    page_offsets: list[int] = field(default_factory=list)
    """Character offset in ``text`` at which each page begins."""
    page_count: int = 0
    pages_extracted: int = 0
    truncated: bool = False

    def page_of_offset(self, offset: int) -> int | None:
        """1-based page containing a character offset."""
        if not self.page_offsets or offset < 0:
            return None
        page = 0
        for index, start in enumerate(self.page_offsets):
            if start <= offset:
                page = index + 1
            else:
                break
        return page or None


def looks_like_pdf(body: bytes, content_type: str = "") -> bool:
    """Detect a PDF by magic bytes first, content-type second.

    Extension is not consulted at all: plenty of PDFs are served from URLs
    with no suffix, and plenty of ``.pdf`` URLs return an HTML error page.
    """
    if body[:5] == PDF_MAGIC:
        return True
    # Some servers emit a UTF-8 BOM or stray whitespace before the header.
    if PDF_MAGIC in body[:1024] and b"<html" not in body[:1024].lower():
        return True
    return "application/pdf" in content_type.lower()


def extract_pdf(
    body: bytes,
    *,
    max_pages: int = 60,
    max_chars: int = 200_000,
) -> PdfDocument:
    """Extract text from PDF bytes, preserving page boundaries.

    Raises :class:`PdfExtractionError` for a file that cannot be parsed and
    :class:`ScannedPdfError` for one that parses but yields no usable text.
    Caps are applied because a thousand-page PDF is a memory and token
    problem, not a research opportunity.
    """
    try:
        reader = PdfReader(io.BytesIO(body), strict=False)
    except (PdfReadError, PdfStreamError, ValueError, OSError) as exc:
        raise PdfExtractionError(f"unreadable PDF: {exc}") from exc

    try:
        if reader.is_encrypted:
            # An empty user password is common for "protected" documents and
            # is worth one attempt before giving up.
            try:
                reader.decrypt("")
            except Exception as exc:
                raise PdfExtractionError(f"encrypted PDF: {exc}") from exc
    except PdfReadError as exc:
        raise PdfExtractionError(f"encrypted PDF: {exc}") from exc

    try:
        page_count = len(reader.pages)
    except (PdfReadError, PdfStreamError, ValueError) as exc:
        raise PdfExtractionError(f"unreadable page tree: {exc}") from exc
    if page_count == 0:
        raise PdfExtractionError("PDF contains no pages")

    chunks: list[str] = []
    offsets: list[int] = []
    total = 0
    extracted = 0
    truncated = page_count > max_pages

    for index, page in enumerate(reader.pages[:max_pages]):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:
            log.debug("pdf_page_failed", page=index + 1, error=str(exc)[:120])
            raw = ""
        text = clean_text(raw)
        # The offset is recorded even for an empty page so page numbering
        # stays aligned with the document rather than with extraction luck.
        offsets.append(total)
        if not text:
            continue
        if total + len(text) > max_chars:
            text = text[: max_chars - total]
            truncated = True
        chunks.append(text)
        total += len(text) + 1  # the newline joined in below
        extracted += 1
        if total >= max_chars:
            truncated = True
            break

    combined = "\n".join(chunks).strip()
    if not combined or len(combined) < _MIN_CHARS_PER_PAGE * min(page_count, 3):
        raise ScannedPdfError(
            f"PDF yielded {len(combined)} characters across {page_count} page(s); "
            "it is probably scanned images. OCR is not supported."
        )

    return PdfDocument(
        text=combined,
        page_offsets=offsets,
        page_count=page_count,
        pages_extracted=extracted,
        truncated=truncated,
    )
