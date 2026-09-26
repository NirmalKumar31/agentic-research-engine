"""A deterministic stand-in for the NLI model, for tests.

CI does not download a 1.4GB checkpoint on every push, so the adapter,
the guards and the publication gate are exercised against fixed scores
here. This measures wiring, not semantics: no calibration number may be
derived from it. The real model runs in the dedicated semantic
calibration workflow and in the release gate.
"""

from __future__ import annotations

from agentic_research.citations.nli import NLIPrediction, NLIScores, NLIUnavailable


class FakeScorer:
    """Returns pre-set scores, keyed by (premise, hypothesis)."""

    model_id = "fake/deterministic"
    revision = "test"

    def __init__(
        self,
        scores: dict[tuple[str, str], tuple[float, float, float]] | None = None,
        *,
        default: tuple[float, float, float] = (0.0, 1.0, 0.0),
        fail_with: str | None = None,
    ) -> None:
        self._scores = scores or {}
        self._default = default
        self._fail_with = fail_with
        self.calls = 0
        self.seen: list[tuple[str, str]] = []
        """Every (premise, hypothesis) pair handed to the scorer.

        Recorded so tests can assert what the classifier was actually
        shown -- specifically that no source identity, quality score or
        ranking reached the premise."""

    def score(self, pairs: list[tuple[str, str]]) -> list[NLIPrediction]:
        self.calls += 1
        self.seen.extend(pairs)
        if self._fail_with is not None:
            raise NLIUnavailable(self._fail_with)
        out = []
        for premise, hypothesis in pairs:
            e, n, c = self._scores.get((premise, hypothesis), self._default)
            out.append(
                NLIPrediction(
                    premise=premise,
                    hypothesis=hypothesis,
                    scores=NLIScores(entailment=e, neutral=n, contradiction=c),
                    model_id=self.model_id,
                    model_revision=self.revision,
                )
            )
        return out


class BrokenScorer:
    """Returns the wrong number of results, or out-of-range probabilities."""

    model_id = "fake/broken"
    revision = "test"

    def __init__(self, *, mode: str = "short") -> None:
        self.mode = mode

    def score(self, pairs: list[tuple[str, str]]) -> list[NLIPrediction]:
        if self.mode == "short":
            pairs = pairs[:-1] if len(pairs) > 1 else []
        value = 1.5 if self.mode == "out_of_range" else 0.99
        return [
            NLIPrediction(
                premise=p,
                hypothesis=h,
                scores=NLIScores(entailment=value, neutral=0.0, contradiction=0.0),
                model_id=self.model_id,
                model_revision=self.revision,
            )
            for p, h in pairs
        ]
