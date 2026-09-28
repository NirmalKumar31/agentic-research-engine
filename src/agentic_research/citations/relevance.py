"""Whether a claim answers the question, which is not whether it is true.

Every gate in this project asks the same thing from a different angle:
does the evidence carry the claim? None of them asks whether the claim
has anything to do with what was asked. So a live run answered "how
does a large language model differ from a neural network" with five
well-cited claims describing what a large language model is, and a
fixture answers a question about PostgreSQL with a MySQL benchmark at
entailment 0.999. Entailment cannot catch either. It is not supposed
to.

This is the missing question, asked separately and answered against
the contract built before retrieval.

Three signals, deliberately not one:

*Contract alignment* is deterministic. The claim declares which slot
it fills; that slot must exist, and a definition cannot fill a
contrast.

*Structure* is deterministic. A claim about the wrong entity, or the
wrong period, is not about the question however well supported.

*A separate judgement* is asked of a model that did not propose the
claim, because a model marking its own homework will find it relevant.
It is required where configured and its absence withholds -- an
unanswered relevance question is not a "yes".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentic_research.answer_contract import AnswerContract, QuestionType

# Words that make a sentence a comparison rather than a description.
# Deliberately small: this decides whether a claim can fill a contrast
# slot, and a broad list would let "LLMs are also neural networks"
# count as a contrast.
_CONTRAST_MARKERS = (
    "differ",
    "differs",
    "difference",
    "unlike",
    "whereas",
    "compared with",
    "compared to",
    "in contrast",
    "rather than",
    "more than",
    "less than",
    "faster than",
    "slower than",
    "larger than",
    "smaller than",
    "instead of",
)

_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")


@dataclass(frozen=True)
class RelevanceVerdict:
    """Why a claim does or does not answer the question."""

    answers_question: bool
    answer_slot: str | None
    slot_satisfied: bool
    reason: str
    checked: bool = True
    entity_aligned: bool = True
    period_aligned: bool = True
    failures: tuple[str, ...] = field(default_factory=tuple)

    @property
    def publishable(self) -> bool:
        return self.checked and self.answers_question and self.slot_satisfied


def _no(reason: str, slot: str | None, *, checked: bool = True, **flags: bool) -> RelevanceVerdict:
    return RelevanceVerdict(
        answers_question=False,
        answer_slot=slot,
        slot_satisfied=False,
        reason=reason,
        checked=checked,
        entity_aligned=flags.get("entity_aligned", True),
        period_aligned=flags.get("period_aligned", True),
        failures=tuple(k for k, v in flags.items() if v is False) or (reason,),
    )


def _mentions(text: str, term: str) -> bool:
    """Whether the text is about this subject.

    Four ways, tried in order, each looser than the last. The order
    matters: the loosest rule alone would match "vector database" to a
    claim about vectors, and the strictest alone would reject a paper
    that answers a question about self-attention while only ever
    writing "scaled dot-product attention".

    1. the phrase itself, allowing a plural;
    2. its acronym, so "LLM" answers "large language model";
    3. its head noun, so "attention" answers "self-attention" and
       "neural networks" answers "general neural network";
    4. its distinctive stems, so "rotating a key" answers "API key
       rotation" -- but two-stem terms need both, so "vectors" alone
       does not answer "vector database".
    """
    lowered = text.lower()
    term = term.strip().lower()
    if not term:
        return False

    def present(needle: str) -> bool:
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?:e?s)?(?![a-z0-9])", lowered))

    if present(term):
        return True

    words = [w for w in re.split(r"[^a-z0-9]+", term) if w]
    if len(words) < 2:
        return False

    if present("".join(w[0] for w in words)):
        return True

    # Head noun: the last word, and the last two for longer terms.
    if present(words[-1]):
        return True
    if len(words) > 2 and present(" ".join(words[-2:])):
        return True

    # Stems, as a last resort. A single distinctive stem is enough;
    # two or more must nearly all appear, or any claim mentioning
    # "vectors" would count as being about vector databases.
    stems = [w[:4] for w in words if len(w) >= 4]
    if not stems:
        return False
    hits = sum(1 for stem in stems if re.search(rf"(?<![a-z0-9]){stem}", lowered))
    required = 1 if len(stems) == 1 else max(2, len(stems) - 1)
    return hits >= required


def _is_contrastive(text: str, entities: tuple[str, ...]) -> bool:
    """Both subjects named, and something said about the difference.

    Naming both is not enough: "LLMs are neural networks" names both
    and states no contrast. A marker without both subjects is not
    enough either.
    """
    lowered = text.lower()
    named = sum(1 for e in entities if _mentions(text, e))
    has_marker = any(m in lowered for m in _CONTRAST_MARKERS)
    return named >= 2 and has_marker


def _competing_subject(claim_text: str, contract: AnswerContract) -> str | None:
    """A named subject the question never mentions, if there is one.

    Sentence-initial words are skipped: every sentence starts with a
    capital and that says nothing. A token counts when it is
    capitalised mid-sentence, or carries internal capitals the way
    MySQL and PostgreSQL do.
    """
    haystack = " ".join([contract.question, *contract.entities]).lower()
    sentences = re.split(r"(?<=[.!?])\s+", claim_text.strip())
    for sentence in sentences:
        tokens = sentence.split()
        for position, token in enumerate(tokens):
            word = token.strip(".,;:()[]\"'")
            if len(word) < 3:
                continue
            internal_caps = any(c.isupper() for c in word[1:])
            # Every sentence starts with a capital, so position alone
            # says nothing there -- but MySQL is a name wherever it
            # appears, and the wrong-entity case puts it first.
            looks_named = internal_caps or (position > 0 and word[0].isupper())
            if not looks_named:
                continue
            if word.lower() in haystack:
                continue
            # A word the question uses in another form is not a
            # different subject.
            if _mentions(haystack, word):
                continue
            return word
    return None


def assess_relevance(
    claim_text: str,
    declared_slot: str | None,
    contract: AnswerContract,
    *,
    evidence_text: str = "",
    model_says_relevant: bool | None = None,
) -> RelevanceVerdict:
    """Whether this claim answers the question the contract describes.

    ``model_says_relevant`` is the separate judgement. ``None`` means
    it was not obtained; that withholds, because an unanswered
    relevance question is not a "yes".
    """
    if not contract.usable:
        return _no(
            f"no answer contract: {contract.unusable_reason}",
            declared_slot,
            checked=False,
        )

    if not declared_slot:
        return _no("the claim declares no answer slot", None)

    if not contract.has_slot(declared_slot):
        return _no(
            f"slot {declared_slot!r} is not required by this question "
            f"(expected one of {sorted(contract.slot_names)})",
            declared_slot,
        )

    # Entity alignment, as a negative check rather than a requirement.
    #
    # Requiring the claim to name the question's subject was the
    # obvious rule and it is wrong: a source may answer correctly in
    # different words. A question about self-attention is answered by
    # a paper that only ever writes "scaled dot-product attention",
    # and demanding the question's vocabulary rejected the one claim
    # that answered it.
    #
    # What can be caught deterministically is the opposite: a claim
    # about a *different named subject*. A MySQL benchmark answering a
    # question about PostgreSQL is supported at 0.999 and is not an
    # answer. So the rule is -- if the claim names none of the
    # question's subjects and does name some other proper subject the
    # question never mentions, it is about something else.
    #
    # A heuristic, and deliberately a narrow one. The independent
    # judgement below is what catches the rest.
    if contract.entities and not any(_mentions(claim_text, e) for e in contract.entities):
        intruder = _competing_subject(claim_text, contract)
        detail = f"it is about {intruder}" if intruder else "it names none of them, in any wording"
        return _no(
            f"the question asks about {', '.join(contract.entities)} and {detail}",
            declared_slot,
            entity_aligned=False,
        )

    # Period alignment, only when the question fixed a period and the
    # claim names one. A claim naming no year is not thereby wrong.
    contract_years = {y for c in contract.constraints for y in _YEAR.findall(c)}
    contract_years |= set(_YEAR.findall(contract.question))
    if contract_years:
        claim_years = set(_YEAR.findall(claim_text)) | set(_YEAR.findall(evidence_text))
        if claim_years and not (claim_years & contract_years):
            return _no(
                f"the claim is about {sorted(claim_years)} and the question asks "
                f"about {sorted(contract_years)}",
                declared_slot,
                period_aligned=False,
            )

    # A contrast slot needs an actual contrast. This is the live
    # failure: five definitions, none of them a difference.
    if (
        contract.question_type is QuestionType.COMPARISON
        and declared_slot == "direct_contrast"
        and not _is_contrastive(claim_text, contract.entities)
    ):
        return _no(
            "the claim describes one subject rather than contrasting them, "
            "so it cannot fill the contrast slot",
            declared_slot,
        )

    if model_says_relevant is None:
        return _no(
            "no independent relevance judgement was available",
            declared_slot,
            checked=False,
        )
    if model_says_relevant is False:
        return _no("the independent relevance judgement rejected it", declared_slot)

    return RelevanceVerdict(
        answers_question=True,
        answer_slot=declared_slot,
        slot_satisfied=True,
        reason=f"fills the {declared_slot} slot",
    )
