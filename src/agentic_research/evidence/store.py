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
    QuoteMatch,
    SourceDocument,
    Stance,
    SubQuestion,
    SubQuestionCoverage,
)
from agentic_research.retrieval.urls import domain_of

# Similarity at or above which a non-exact quote is recorded as FUZZY.
# Fuzzy matches are diagnostic only: they are never citable, so this threshold
# controls what gets *reported* as drift, not what gets trusted.
_FUZZY_THRESHOLD = 0.88

# Shorter spans match by coincidence and prove nothing.
_MIN_QUOTE_CHARS = 12

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


def classify_quote(quote: str, source_text: str) -> tuple[QuoteMatch, int | None]:
    """Classify how well a quote aligns with its source, and where.

    Returns the match class and the character offset of the match in the
    source (used to recover a PDF page number), or ``None`` when unlocated.

    Three classes rather than a boolean, because the previous single flag
    accepted a 0.88 similarity match and the result was described as
    "verbatim". It is not. ``EXACT_NORMALIZED`` permits whitespace and
    smart-punctuation normalisation and nothing else; anything looser is
    ``FUZZY``, kept for diagnostics but never citable.
    """
    if not quote.strip() or not source_text.strip():
        return QuoteMatch.NONE, None

    needle = _normalise_for_match(quote)
    haystack = _normalise_for_match(source_text)
    if len(needle) < _MIN_QUOTE_CHARS:
        # Too short to be evidence of anything; a common word would match.
        return QuoteMatch.NONE, None

    exact_at = haystack.find(needle)
    if exact_at >= 0:
        return QuoteMatch.EXACT_NORMALIZED, _offset_in_original(source_text, exact_at)

    window = len(needle)
    if window > len(haystack):
        ratio = SequenceMatcher(None, needle, haystack).ratio()
        return (QuoteMatch.FUZZY if ratio >= _FUZZY_THRESHOLD else QuoteMatch.NONE), None

    # Sliding comparison, stepped at a quarter of the quote length. Cheap
    # enough for page-sized text and tolerant of small edits.
    step = max(1, window // 4)
    matcher = SequenceMatcher()
    matcher.set_seq2(needle)
    for start_at in range(0, len(haystack) - window + 1, step):
        matcher.set_seq1(haystack[start_at : start_at + window])
        if matcher.quick_ratio() < _FUZZY_THRESHOLD:
            continue
        if matcher.ratio() >= _FUZZY_THRESHOLD:
            return QuoteMatch.FUZZY, _offset_in_original(source_text, start_at)
    return QuoteMatch.NONE, None


def _offset_in_original(source_text: str, normalised_offset: int) -> int:
    """Map an offset in the normalised text back to the original.

    Normalisation only collapses whitespace, so walking both strings in step
    recovers the original position closely enough to identify a PDF page.
    """
    original = 0
    normalised = 0
    previous_space = False
    for character in source_text:
        if normalised >= normalised_offset:
            break
        is_space = character.isspace()
        if is_space and previous_space:
            original += 1
            continue
        normalised += 1
        original += 1
        previous_space = is_space
    return original


def verify_quote(quote: str, source_text: str) -> bool:
    """Whether a quote is genuinely present in the source.

    Exact-normalised only. Kept as a convenience wrapper over
    :func:`classify_quote` for call sites that only need the boolean.
    """
    match, _ = classify_quote(quote, source_text)
    return match is QuoteMatch.EXACT_NORMALIZED


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
    evidence_ids: list[str]
    evidence_count: int
    dropped_low_confidence: int


class EvidenceStore:
    """Read-only index over the sources and evidence gathered so far."""

    def __init__(self, sources: list[SourceDocument], evidence: list[EvidenceItem]) -> None:
        self._sources = sources
        self._evidence = evidence
        self._by_source_id = {s.id: s for s in sources}
        self._by_evidence_id = {e.id: e for e in evidence}

    # -- lookups -----------------------------------------------------------

    @property
    def sources(self) -> list[SourceDocument]:
        return self._sources

    @property
    def evidence(self) -> list[EvidenceItem]:
        return self._evidence

    def source(self, source_id: str) -> SourceDocument | None:
        return self._by_source_id.get(source_id)

    def evidence_by_id(self, evidence_id: str) -> EvidenceItem | None:
        """Resolve an evidence id. The engine's half of claim provenance."""
        return self._by_evidence_id.get(evidence_id)

    def citable_evidence(self) -> list[EvidenceItem]:
        """Evidence eligible to ground a claim in the final report."""
        return [e for e in self._evidence if e.is_citable]

    def resolve_citations(self, evidence_ids: list[str]) -> list[str]:
        """Derive source ids from evidence ids, preserving order.

        This is the engine performing the evidence -> source resolution that
        a model must not be trusted to do for itself; letting the model emit
        both invites the two to disagree."""
        sources: list[str] = []
        for evidence_id in evidence_ids:
            item = self._by_evidence_id.get(evidence_id)
            if item is not None and item.source_id not in sources:
                sources.append(item.source_id)
        return sources

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
            note = f"{len(verified)} exact-match items across {distinct_sources} sources"
        elif verified:
            verdict = "weak"
            note = f"only {len(verified)} exact-match item(s) from {distinct_sources} source(s)"
        else:
            verdict = "uncovered"
            note = "no exact-match evidence"

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
        citable_only: bool = True,
    ) -> EvidencePackage:
        """Render the evidence into the prompt block used for synthesis.

        ``citable_only`` defaults to True: synthesis sees only evidence whose
        quote was located in the source. Previously a fuzzy or unlocatable
        quote could still clear the confidence floor (0.7 relevance x 0.4
        penalty = 0.28 > 0.25) and go on to ground a citation, meaning a claim
        could rest entirely on text nobody could find in the page. Diagnostic
        callers pass False to inspect everything that was extracted.
        """
        lines: list[str] = []
        used_source_ids: list[str] = []
        used_evidence_ids: list[str] = []
        included = 0
        dropped = 0

        for sub_question in sub_questions:
            items = self.for_sub_question(sub_question.id)
            if not items:
                continue
            eligible = [e for e in items if e.is_citable] if citable_only else items
            kept = [e for e in eligible if e.confidence >= min_confidence]
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
                used_evidence_ids.append(item.id)
                quote = item.quote.strip()
                if len(quote) > max_quote_chars:
                    quote = quote[:max_quote_chars].rstrip() + "..."
                page = f", p. {item.page}" if item.page else ""
                # The evidence id is what the synthesiser must reference.
                # Showing the source id here would invite it to cite sources
                # directly and reintroduce the ambiguity this design removes.
                lines.append(
                    f'- {item.id} ({item.stance.value}{page}) {item.claim}\n  quote: "{quote}"'
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
            evidence_ids=used_evidence_ids,
            evidence_count=included,
            dropped_low_confidence=dropped,
        )
