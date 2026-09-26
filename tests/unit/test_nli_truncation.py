"""A truncated premise is not the premise.

A quote whose opening supports a claim and whose closing qualifies it
away scores as support once the tail is cut, and nothing in the score
reveals it. Real evidence quotes are far shorter than the model's
context today — the longest across all three canonical recordings is
122 tokens against a 512 limit — but "currently short enough" is not a
guarantee, so overlong pairs fail closed instead.
"""

from __future__ import annotations

import pytest

from agentic_research.citations.fake_nli import FakeScorer
from agentic_research.citations.guards import SourceIdentity
from agentic_research.citations.nli import NLIPrediction, NLIScores
from agentic_research.citations.semantic import CitedEvidence, verify_claim


class TruncatingScorer:
    """Reports that it had to cut the pair, and scores it as support."""

    model_id = "fake/truncating"
    revision = "test"

    def score(self, pairs: list[tuple[str, str]]) -> list[NLIPrediction]:
        return [
            NLIPrediction(
                premise=p,
                hypothesis=h,
                scores=NLIScores(entailment=0.999, neutral=0.001, contradiction=0.0),
                model_id=self.model_id,
                model_revision=self.revision,
                truncated=True,
            )
            for p, h in pairs
        ]


QUOTE = (
    "The system reached 99.9% uptime in the trial period. "
    + "Filler sentence to push the qualifier past the boundary. " * 40
    + "However, the trial excluded the two largest regions."
)


class TestTruncatedPremisesDoNotPublish:
    def test_a_truncated_pair_is_not_usable_support(self) -> None:
        verdict = verify_claim(
            "The system reached 99.9% uptime.",
            [CitedEvidence("E1", QUOTE, SourceIdentity("example.test", "A Page"))],
            TruncatingScorer(),
            support_threshold=0.98,
        )
        assert not verdict.publishable, verdict.reason

    def test_the_reason_names_truncation(self) -> None:
        verdict = verify_claim(
            "The system reached 99.9% uptime.",
            [CitedEvidence("E1", QUOTE)],
            TruncatingScorer(),
            support_threshold=0.98,
        )
        assert "premise-truncated" in verdict.per_evidence[0].failed_guard_names

    def test_it_withholds_even_at_maximum_entailment(self) -> None:
        """The score is meaningless if the model never saw the qualifier,
        so no threshold can rescue it."""
        verdict = verify_claim(
            "The system reached 99.9% uptime.",
            [CitedEvidence("E1", QUOTE)],
            TruncatingScorer(),
            support_threshold=0.0,
        )
        assert not verdict.publishable

    def test_an_untruncated_pair_is_unaffected(self) -> None:
        verdict = verify_claim(
            "Throughput rose 20%.",
            [CitedEvidence("E1", "Throughput rose 20% after the change.")],
            FakeScorer(default=(0.99, 0.01, 0.0)),
            support_threshold=0.98,
        )
        assert verdict.publishable

    def test_a_truncated_item_cannot_be_the_best_evidence(self) -> None:
        """A cut quote scoring 0.999 must not outrank an intact one."""

        class Mixed:
            model_id = "fake/mixed"
            revision = "test"

            def score(self, pairs: list[tuple[str, str]]) -> list[NLIPrediction]:
                out = []
                for p, h in pairs:
                    cut = p.startswith("LONG")
                    out.append(
                        NLIPrediction(
                            premise=p,
                            hypothesis=h,
                            scores=NLIScores(
                                entailment=0.999 if cut else 0.50,
                                neutral=0.001 if cut else 0.50,
                                contradiction=0.0,
                            ),
                            model_id=self.model_id,
                            model_revision=self.revision,
                            truncated=cut,
                        )
                    )
                return out

        verdict = verify_claim(
            "Throughput rose 20%.",
            [
                CitedEvidence("E1", "LONG cut quote about throughput"),
                CitedEvidence("E2", "Throughput rose 20%."),
            ],
            Mixed(),
            support_threshold=0.98,
        )
        assert not verdict.publishable
        assert verdict.best_evidence_id == "E2"


class TestRealQuotesFitTheContext:
    @pytest.mark.nli
    def test_no_recorded_quote_approaches_the_limit(self) -> None:
        """Measured, not assumed. If a future extractor starts emitting
        long quotes this fails before a truncated premise reaches the
        gate."""
        import json
        from pathlib import Path

        from agentic_research.citations.nli import _load
        from agentic_research.config import Settings

        settings = Settings()
        tokenizer, _model, _labels = _load(settings.nli_model_id, settings.nli_model_revision)
        longest = 0
        recordings = Path("src/agentic_research/web/recorded_runs")
        for path in sorted(recordings.glob("*.json")):
            for item in json.loads(path.read_text())["result"]["evidence"]:
                tokens = tokenizer(item.get("quote") or "", add_special_tokens=True)["input_ids"]
                longest = max(longest, len(tokens))
        assert longest < 512, f"longest recorded quote is {longest} tokens"
