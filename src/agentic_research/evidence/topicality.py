"""Whether a piece of evidence is about the sub-question it was filed under.

``quote_verified`` means the quote appears verbatim in the source that
cites it. It is a provenance property and says nothing about subject
matter, but coverage treated it as sufficient: two exact quotes from two
sources marked a sub-question covered.

A live run showed what that permits. Asked for the main causes of
overfitting, retrieval returned papers on double descent and frozen
overparameterization; the extractor pulled exact quotes from them and
attributed them to sub-questions about training-data size and data
leakage; coverage counted those sub-questions answered. Nothing had
addressed them.

The test here is lexical and deliberately weak -- shared salient terms,
not a judgement about whether the evidence answers anything. Three
reasons for keeping it weak:

* a strict test would starve coverage and make runs worse, and this
  module exists to stop a false positive, not to create false
  negatives;
* the relevance judge downstream is what decides whether a *claim*
  answers the question, and it has the contract to judge against;
* it must stay deterministic and free, because it runs per evidence
  item per sub-question and a model call there would multiply cost.
"""

from __future__ import annotations

import re

# Words that carry no subject matter. Kept small on purpose: a long
# stoplist starts removing terms that do distinguish one sub-question
# from another ("training", "data", "model" are the subject here).
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "against",
        "also",
        "another",
        "any",
        "are",
        "because",
        "been",
        "being",
        "between",
        "both",
        "can",
        "could",
        "did",
        "does",
        "doing",
        "during",
        "each",
        "from",
        "further",
        "had",
        "has",
        "have",
        "how",
        "into",
        "its",
        "itself",
        "just",
        "may",
        "might",
        "more",
        "most",
        "much",
        "must",
        "not",
        "now",
        "only",
        "other",
        "our",
        "out",
        "over",
        "own",
        "same",
        "should",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "under",
        "until",
        "very",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "would",
        "your",
    }
)

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")

# Two shared terms rather than one. One is met by coincidence -- almost
# any machine-learning text shares "data" with almost any other -- and
# that coincidence is exactly what let double-descent papers count as
# evidence about training-set size.
_REQUIRED_OVERLAP = 2


def salient_terms(text: str) -> frozenset[str]:
    """Lower-cased content words of three characters or more."""
    return frozenset(
        token.lower() for token in _TOKEN.findall(text or "") if token.lower() not in _STOPWORDS
    )


def overlap(question: str, evidence: str) -> int:
    """How many salient terms the two share."""
    return len(salient_terms(question) & salient_terms(evidence))


def is_topical(question: str, evidence: str) -> bool:
    """Whether this evidence is plausibly about this sub-question.

    The two empty cases are deliberately *not* symmetric.

    A sub-question with no salient terms gives this test nothing to
    compare against, so it abstains and the item passes. Rejecting
    there would call evidence off-topic without knowing the topic --
    punishing a run for a degenerate question rather than for anything
    the evidence did.

    Evidence with no salient terms is rejected: an item with no content
    words cannot be shown to address anything, and this is the one
    direction where abstaining would reinstate the defect for exactly
    the evidence that carries least.

    A sub-question with fewer salient terms than the threshold is held
    to what it has, so a short question is not automatically
    unanswerable.
    """
    terms = salient_terms(question)
    if not terms:
        return True
    if not salient_terms(evidence):
        return False
    return overlap(question, evidence) >= min(_REQUIRED_OVERLAP, len(terms))
