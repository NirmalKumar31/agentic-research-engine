"""The publication decision: NLI probabilities plus deterministic guards.

The gate is binary. A claim either passed high-confidence semantic
verification or it did not, and only the first publishes. The three-way
verdict the rest of the system still speaks in is derived from the
scores afterwards, for diagnostics; it does not control anything.

Structure of the decision, in order:

1. Every cited evidence quote is scored against the claim, separately.
   Quotes are never concatenated -- joining eight unrelated snippets
   until they collectively imply something is exactly the reasoning the
   verifier is supposed to prevent.
2. An evidence item is *valid* for a claim only if the deterministic
   guards pass against that item's own text.
3. The claim publishes when some valid item entails it at or above the
   support threshold.

Everything else withholds, including every failure path. A claim that
could not be checked is indistinguishable from a claim that failed, and
both must stay out of the report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal, Protocol

from agentic_research.citations.guards import (
    GuardResult,
    SourceIdentity,
    guards_pass,
    run_guards,
)
from agentic_research.citations.nli import NLIPrediction, NLIUnavailable

if TYPE_CHECKING:
    from agentic_research.models import ClaimJudgment, ClaimKind

# Diagnostic verdicts. Only SUPPORTED publishes; the boundary between
# the other two is informational and deliberately not release-critical.
Verdict = Literal["supported", "partially_supported", "unsupported"]

SUPPORTED: Final[Verdict] = "supported"
PARTIALLY_SUPPORTED: Final[Verdict] = "partially_supported"
UNSUPPORTED: Final[Verdict] = "unsupported"

# A claim the evidence actively argues against, rather than merely
# failing to establish. Diagnostic only.
_CONTRADICTION_LEVEL = 0.5
# Below this the evidence says essentially nothing about the claim.
_NEGLIGIBLE_ENTAILMENT = 0.05


class Scorer(Protocol):
    """What the decision needs from a scorer.

    Narrow on purpose: CI substitutes a deterministic fake rather than
    downloading a checkpoint, and the calibration harness swaps models
    without touching this file.
    """

    model_id: str
    revision: str

    def score(self, pairs: list[tuple[str, str]]) -> list[NLIPrediction]: ...


@dataclass(frozen=True)
class CitedEvidence:
    """One cited quote, plus who published it.

    ``source`` reaches the deterministic attribution guard and stops
    there. It is never part of the NLI premise: a classifier told that
    a quote came from an authoritative domain would be scoring
    reputation, and entailment is the only thing it is allowed to score.
    """

    evidence_id: str
    quote: str
    source: SourceIdentity | None = None


@dataclass(frozen=True)
class EvidenceScore:
    evidence_id: str
    entailment: float
    neutral: float
    contradiction: float
    guards: list[GuardResult]
    truncated: bool = False

    @property
    def guards_passed(self) -> bool:
        """Usable as support. A truncated premise is not the premise.

        A quote whose opening supports a claim and whose tail qualifies
        it away would score as support once the tail is cut, and nothing
        in the score would show it. Treated here as a failed check
        rather than a lower number.
        """
        return not self.truncated and guards_pass(self.guards)

    @property
    def failed_guard_names(self) -> list[str]:
        names = [g.name for g in self.guards if not g.passed]
        if self.truncated:
            names.append("premise-truncated")
        return names


@dataclass(frozen=True)
class SemanticVerdict:
    """The complete, untruncated record of why a claim did or did not publish."""

    publishable: bool
    verdict: Verdict
    reason: str
    checked: bool
    support_threshold: float
    model_id: str
    model_revision: str
    best_evidence_id: str | None = None
    best_entailment: float = 0.0
    per_evidence: list[EvidenceScore] = field(default_factory=list)


def _withheld(
    reason: str, threshold: float, scorer_id: str, revision: str, *, checked: bool
) -> SemanticVerdict:
    return SemanticVerdict(
        publishable=False,
        verdict=UNSUPPORTED,
        reason=reason,
        checked=checked,
        support_threshold=threshold,
        model_id=scorer_id,
        model_revision=revision,
    )


def verify_claim(
    claim_text: str,
    evidence: Sequence[CitedEvidence | tuple[str, str]],
    scorer: Scorer,
    *,
    support_threshold: float,
) -> SemanticVerdict:
    """Decide whether one claim may publish.

    ``evidence`` is the cited, citable items. An empty list withholds: a
    claim with nothing to check against has not passed verification,
    whatever the reason it ended up that way.

    Plain ``(evidence_id, quote)`` pairs are accepted for callers with
    no source identity to offer, such as the adversarial suite; those
    simply leave the attribution guard nothing to check against.
    """
    cited: list[CitedEvidence] = [
        item if isinstance(item, CitedEvidence) else CitedEvidence(item[0], item[1])
        for item in evidence
    ]
    if not cited:
        return _withheld(
            "no citable evidence resolved for this claim",
            support_threshold,
            scorer.model_id,
            scorer.revision,
            checked=False,
        )

    try:
        # The premise is the exact quote and nothing else -- no source
        # title, domain, category, rank or quality score.
        predictions = scorer.score([(item.quote, claim_text) for item in cited])
    except NLIUnavailable as exc:
        # Deliberately not a fallback to any other verifier. An
        # unavailable classifier means unverified, and unverified means
        # withheld.
        return _withheld(
            f"semantic verification unavailable: {exc}",
            support_threshold,
            scorer.model_id,
            scorer.revision,
            checked=False,
        )

    if len(predictions) != len(cited):
        return _withheld(
            f"scorer returned {len(predictions)} results for {len(cited)} pairs",
            support_threshold,
            scorer.model_id,
            scorer.revision,
            checked=False,
        )

    scores: list[EvidenceScore] = []
    for item, prediction in zip(cited, predictions, strict=True):
        evidence_id, quote = item.evidence_id, item.quote
        s = prediction.scores
        if not all(0.0 <= v <= 1.0 for v in (s.entailment, s.neutral, s.contradiction)):
            return _withheld(
                f"malformed scores for {evidence_id}",
                support_threshold,
                scorer.model_id,
                scorer.revision,
                checked=False,
            )
        scores.append(
            EvidenceScore(
                evidence_id=evidence_id,
                entailment=s.entailment,
                neutral=s.neutral,
                contradiction=s.contradiction,
                guards=run_guards(claim_text, quote, item.source),
                truncated=getattr(prediction, "truncated", False),
            )
        )

    model_id, revision = scorer.model_id, scorer.revision

    # Publication looks only at items that survived the guards. The
    # highest-entailment item overall may well be one whose numbers or
    # modality do not match, and it must not carry the claim.
    valid = [s for s in scores if s.guards_passed]
    best_valid = max(valid, key=lambda s: s.entailment, default=None)
    best_any = max(scores, key=lambda s: s.entailment)

    if best_valid is not None and best_valid.entailment >= support_threshold:
        return SemanticVerdict(
            publishable=True,
            verdict=SUPPORTED,
            reason=(
                f"entailed by {best_valid.evidence_id} at "
                f"{best_valid.entailment:.3f} >= {support_threshold:.2f}"
            ),
            checked=True,
            support_threshold=support_threshold,
            model_id=model_id,
            model_revision=revision,
            best_evidence_id=best_valid.evidence_id,
            best_entailment=best_valid.entailment,
            per_evidence=scores,
        )

    if not valid:
        failed = sorted({name for s in scores for name in s.failed_guard_names})
        reason = f"every cited quote failed a deterministic guard: {', '.join(failed)}"
    else:
        assert best_valid is not None
        reason = (
            f"best entailment {best_valid.entailment:.3f} from {best_valid.evidence_id} "
            f"is below the {support_threshold:.2f} support threshold"
        )

    return SemanticVerdict(
        publishable=False,
        verdict=_diagnostic(scores, best_valid),
        reason=reason,
        checked=True,
        support_threshold=support_threshold,
        model_id=model_id,
        model_revision=revision,
        best_evidence_id=(best_valid or best_any).evidence_id,
        best_entailment=(best_valid or best_any).entailment,
        per_evidence=scores,
    )


def _diagnostic(scores: list[EvidenceScore], best_valid: EvidenceScore | None) -> Verdict:
    """Derive the three-way label after the fact, for humans reading the audit.

    Nothing branches on this. It separates "the evidence argues against
    this" from "the evidence does not carry this", which is useful when
    reading a rejection and useless as a release metric.
    """
    if max((s.contradiction for s in scores), default=0.0) >= _CONTRADICTION_LEVEL:
        return UNSUPPORTED
    if best_valid is None or best_valid.entailment < _NEGLIGIBLE_ENTAILMENT:
        return UNSUPPORTED
    return PARTIALLY_SUPPORTED


def to_judgment(
    claim_text: str,
    kind: ClaimKind,
    evidence_ids: list[str],
    verdict: SemanticVerdict,
) -> ClaimJudgment:
    """Render a verdict as the durable audit record.

    Imported lazily to keep this module free of the Pydantic models, so
    the calibration harness can score cases without pulling in the rest
    of the application.
    """
    from agentic_research.models import ClaimJudgment, EvidenceScoreRecord

    return ClaimJudgment(
        claim_text=claim_text,
        kind=kind,
        evidence_ids=evidence_ids,
        verdict=verdict.verdict,
        reason=verdict.reason,
        checked=verdict.checked,
        model_id=verdict.model_id,
        model_revision=verdict.model_revision,
        support_threshold=verdict.support_threshold,
        evidence_scores=[
            EvidenceScoreRecord(
                evidence_id=s.evidence_id,
                entailment=round(s.entailment, 6),
                neutral=round(s.neutral, 6),
                contradiction=round(s.contradiction, 6),
                guards_passed=s.guards_passed,
                failed_guards=s.failed_guard_names,
            )
            for s in verdict.per_evidence
        ],
        best_evidence_id=verdict.best_evidence_id,
        best_entailment=round(verdict.best_entailment, 6) if verdict.per_evidence else None,
        guards_passed=any(s.guards_passed for s in verdict.per_evidence)
        if verdict.per_evidence
        else None,
        publishable=verdict.publishable,
    )
