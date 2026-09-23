"""Minimal valid PDFs, built by hand.

Generating them in-process keeps the repository free of binary fixtures and
lets a test ask for exactly the shape it needs -- a given page count, a
specific string on page 3, a deliberately truncated file. Written with raw
PDF syntax rather than a writer library so the corpus needs no extra
dependency.
"""

from __future__ import annotations


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def make_pdf(pages: list[str]) -> bytes:
    """Build a valid PDF whose pages contain the given text."""
    objects: list[bytes] = []

    def add(body: str | bytes) -> int:
        objects.append(body.encode("latin-1") if isinstance(body, str) else body)
        return len(objects)

    font_id = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids: list[int] = []
    content_ids: list[int] = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({_escape(text)}) Tj ET"
        content = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
        content_ids.append(add(content))
        page_ids.append(0)  # placeholder, patched below

    pages_id = len(objects) + len(pages) + 1
    for index, content_id in enumerate(content_ids):
        page_ids[index] = add(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        )

    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    actual_pages_id = add(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>")
    catalog_id = add(f"<< /Type /Catalog /Pages {actual_pages_id} 0 R >>")

    # Page objects reference the /Pages object by number, so it must land
    # where we predicted; assert rather than emit a subtly broken file.
    assert actual_pages_id == pages_id, "page tree object id drifted"

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)


def make_scanned_pdf(page_count: int = 2) -> bytes:
    """A PDF that parses but holds no extractable text, like a scan."""
    return make_pdf([""] * page_count)


def make_malformed_pdf() -> bytes:
    """Correct magic bytes, corrupt structure."""
    return b"%PDF-1.7\n" + b"garbage that is not a pdf body " * 20


def make_oversized_pdf(target_bytes: int) -> bytes:
    """A valid, genuinely readable PDF padded past a size cap.

    The text has to be substantial enough to clear the scanned-PDF check, or
    the test measures the wrong thing.
    """
    base = make_pdf(
        [
            "This padding document contains enough real extractable text that "
            "it is not mistaken for a scanned image-only PDF by the extractor.",
            "A second page of ordinary prose, present so the page count and "
            "character count both look like a genuine document.",
        ]
    )
    padding = b"\n% " + b"x" * max(0, target_bytes - len(base) - 4)
    return base + padding
