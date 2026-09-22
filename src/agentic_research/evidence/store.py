"""Evidence indexing, quote verification and synthesis packaging.

The store is a *view* over the lists held in graph state, not a separate
mutable container. Graph state has to stay JSON-serialisable for
checkpointing, and a live object graph in a state channel would either break
that or quietly diverge from the checkpointed copy after a resume. Building a
view costs microseconds and removes the whole class of problem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from agentic_research.models import (
    EvidenceItem,
    SourceDocument,
    Stance,
    SubQuestion,
    SubQuestionCoverage,
)
from agentic_research.retrieval.urls import domain_of

# Below this, a "verbatim" quote is treated as not present in the source.
# Set to tolerate whitespace and punctuation drift but not paraphrase.
_QUOTE_MATCH_THRESHOLD = 0.88

# Smart punctuation is normalised before matching: models routinely retype a
# quote with straight quotes when the page used curly ones, and that should
# not count as a failed verification.
_PUNCT = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)


def _normalise_for_match(text: str) -> str:
    text = text.translate(_PUNCT).lower()
    return re.sub(r"\s+", " ", text).strip()


def verify_quote(quote: str, source_text: str) -> bool:
    """Check that ``quote`` genuinely appears in ``source_text``.

    This is the engine's main defence against a plausible-sounding sentence
    that the page never contained. Models paraphrase when asked to quote, so
    exact matching alone rejects too much; the fallback compares the quote
    against same-length windows of the source and accepts only a very close
    match.
    """
    if not quote.strip() or not source_text.strip():
        return False

    needle = _normalise_for_match(quote)
    haystack = _normalise_for_match(source_text)
    if len(needle) < 12:
        return False
    if needle in haystack:
        return True

    # Sliding comparison, stepped at a quarter of the quote length. Cheap
    # enough for page-sized text and tolerant of small edits.
    window = len(needle)
    if window > len(haystack):
        return SequenceMatcher(None, needle, haystack).ratio() >= _QUOTE_MATCH_THRESHOLD
    step = max(1, window // 4)
    matcher = SequenceMatcher()
    matcher.set_seq2(needle)
    for start in range(0, len(haystack) - window + 1, step):
        matcher.set_seq1(haystack[start : start + window])
        if matcher.quick_ratio() < _QUOTE_MATCH_THRESHOLD:
            continue
        if matcher.ratio() >= _QUOTE_MATCH_THRESHOLD:
            return True
    return False


@dataclass
class EvidencePackage:
    """The curated bundle handed to the synthesiser.

    Synthesis never sees raw pages. It sees verified, attributed, deduplicated
    findings grouped by sub-question. That keeps the prompt small enough to be
    affordable and makes an uncitable claim harder to produce than a citable
    one.
    """

    text: str
    source_ids: list[str]
    evidence_count: int
    dropped_low_confidence: int


class EvidenceStore:
    """Read-only index over the sources and evidence gathered so far."""

    def __init__(self, sources: list[SourceDocument], evidence: list[EvidenceItem]) -> None:
        self._sources = sources
        self._evidence = evidence
        self._by_source_id = {s.id: s for s in sources}

    # -- lookups -----------------------------------------------------------

    @property
    def sources(self) -> list[SourceDocument]:
        return self._sources

    @property
    def evidence(self) -> list[EvidenceItem]:
        return self._evidence

    def source(self, source_id: str) -> SourceDocument | None:
        return self._by_source_id.get(source_id)

    def usable_sources(self) -> list[SourceDocument]:
        return [s for s in self._sources if s.is_usable and not s.duplicate_of]

    def known_source_ids(self) -> set[str]:
        return set(self._by_source_id)

    def for_sub_question(self, sub_question_id: str) -> list[EvidenceItem]:
        return [e for e in self._evidence if e.sub_question_id == sub_question_id]

    def cited_source_ids(self) -> set[str]:
        return {e.source_id for e in self._evidence}

    # -- metrics -----------------------------------------------------------

    def domains(self) -> list[str]:
        return [s.domain or domain_of(s.url) for s in self.usable_sources()]

    def coverage_for(self, sub_question: SubQuestion) -> SubQuestionCoverage:
        """Count-based coverage for one sub-question.

        Deliberately arithmetic. Asking a model "is this covered, 0 to 1"
        produces a number whose meaning nobody can state; counting distinct
        corroborating sources produces one that can be defined in a sentence.
        """
        items = self.for_sub_question(sub_question.id)
        verified = [e for e in items if e.quote_verified]
        distinct_sources = len({e.source_id for e in verified})
        has_contradiction = any(e.stance is Stance.CONTRADICTS for e in items)

        if distinct_sources >= 2 and len(verified) >= 2:
            verdict = "covered"
            note = f"{len(verified)} verified items across {distinct_sources} sources"
        elif verified:
            verdict = "weak"
            note = f"only {len(verified)} verified item(s) from {distinct_sources} source(s)"
        else:
            verdict = "uncovered"
            note = "no verified evidence"

        return SubQuestionCoverage(
            sub_question_id=sub_question.id,
            evidence_count=len(items),
            distinct_sources=distinct_sources,
            has_contradiction=has_contradiction,
            verdict=verdict,
            note=note,
        )

    # -- packaging ---------------------------------------------------------

    def build_package(
        self,
        sub_questions: list[SubQuestion],
        *,
        max_items_per_question: int = 8,
        min_confidence: float = 0.25,
        max_quote_chars: int = 400,
    ) -> EvidencePackage:
        """Render the evidence into the prompt block used for synthesis."""
        lines: list[str] = []
        used_source_ids: list[str] = []
        included = 0
        dropped = 0

        for sub_question in sub_questions:
            items = self.for_sub_question(sub_question.id)
            if not items:
                continue
            kept = [e for e in items if e.confidence >= min_confidence]
            dropped += len(items) - len(kept)
            if not kept:
                continue
            # Contradictions first so they survive the per-question cap; the
            # whole point of tracking disagreement is to not silently drop it.
            kept.sort(key=lambda e: (e.stance is not Stance.CONTRADICTS, -e.confidence))
            kept = kept[:max_items_per_question]

            lines.append(f"\n### {sub_question.id}: {sub_question.text}")
            for item in kept:
                source = self.source(item.source_id)
                if source is None:
                    continue
                if item.source_id not in used_source_ids:
                    used_source_ids.append(item.source_id)
                quote = item.quote.strip()
                if len(quote) > max_quote_chars:
                    quote = quote[:max_quote_chars].rstrip() + "..."
                marker = "" if item.quote_verified else " (quote unverified)"
                lines.append(
                    f"- [{item.source_id}] ({item.stance.value}{marker}) {item.claim}\n"
                    f'  quote: "{quote}"'
                )
                included += 1

        if used_source_ids:
            lines.append("\n### Sources")
            for source_id in used_source_ids:
                source = self.source(source_id)
                if source is None:
                    continue
                date = source.published_date.date().isoformat() if source.published_date else "n.d."
                lines.append(
                    f"[{source_id}] {source.title} — {source.domain} "
                    f"({source.source_type.value}, {date}, quality {source.quality_score:.2f})"
                )

        return EvidencePackage(
            text="\n".join(lines).strip(),
            source_ids=used_source_ids,
            evidence_count=included,
            dropped_low_confidence=dropped,
        )
