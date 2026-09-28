"""Run the frozen adversarial set against the current pipeline.

Credential-free and deterministic. Entailment is pinned per
(claim, evidence) pair in cases.json, because the 0.98 threshold is
frozen and is not what this measures: the subject is evidence
selection, the deterministic guards, relevance to the question, and
whether the report covers what was asked.

Usage:
    python examples/quality-eval/run_eval.py                 # print a report
    python examples/quality-eval/run_eval.py --json out.json # record a baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agentic_research.answer_contract import build_contract
from agentic_research.citations.fake_nli import FakeScorer
from agentic_research.citations.guards import SourceAuthority
from agentic_research.citations.relevance import assess_relevance
from agentic_research.citations.semantic import CitedEvidence, SourceIdentity, verify_claim

HERE = Path(__file__).resolve().parent
THRESHOLD = 0.98


def _scorer(case: dict[str, Any], claim: dict[str, Any]) -> FakeScorer:
    """Scores pinned per pair, so the model is not the variable."""
    quotes = {e["id"]: e["quote"] for e in case["evidence"]}
    scores: dict[tuple[str, str], tuple[float, float, float]] = {}
    for evidence_id, entail in claim["entailment"].items():
        rest = round(1.0 - entail, 6)
        scores[(quotes[evidence_id], claim["text"])] = (entail, rest, 0.0)
    return FakeScorer(scores, default=(0.0, 1.0, 0.0))


def _cited(case: dict[str, Any], claim: dict[str, Any]) -> list[CitedEvidence]:
    quotes = {e["id"]: e for e in case["evidence"]}
    sources = {s["id"]: s for s in case["sources"]}
    out = []
    for evidence_id in claim["cites"]:
        item = quotes[evidence_id]
        src = sources[item["source_id"]]
        out.append(
            CitedEvidence(
                evidence_id,
                item["quote"],
                SourceIdentity(
                    domain=src["domain"],
                    title="",
                    # The fixture's kind and quality reach selection,
                    # never the premise.
                    authority=SourceAuthority(src.get("kind", "unknown")),
                    quality=float(src.get("quality", 0.0)),
                ),
            )
        )
    return out


# The fixtures were written before the contract existed and use their
# own slot vocabulary. The contract is the authority on slot names, so
# the mapping lives here rather than in the fixtures -- editing a
# frozen case to match an implementation is how a baseline stops
# meaning anything.
SLOT_ALIASES: dict[str, str] = {
    "measured_effect": "measured_value",
    "measured_rate": "measured_value",
    "reported_figure": "measured_value",
    "conditions": "measurement_conditions",
    "study_design": "measurement_conditions",
    "source_authority": "limitations",
    "mechanism": "definition",
    "architecture_or_scope": "architecture_or_scope",
    "definition": "part_1",
    "failure_modes": "part_2",
}


def _canonical(slot: str | None, contract: Any) -> str | None:
    if slot is None or contract.has_slot(slot):
        return slot
    mapped = SLOT_ALIASES.get(slot)
    return mapped if mapped and contract.has_slot(mapped) else slot


def _contract(case: dict[str, Any]) -> Any:
    """Rebuild the case's contract through the real constructor."""
    declared = case["contract"]
    qtype = declared["question_type"]
    # Two fixture labels describe the evidence rather than the
    # question, so they are mapped to the question shape they carry.
    qtype = {"insufficient_evidence": "numeric", "terminology_mismatch": "definition"}.get(
        qtype, qtype
    )
    slots = declared["required_slots"]
    dims = [s for s in slots if s not in {"direct_contrast", "relationship", "dimension"}]
    return build_contract(
        case["question"],
        qtype,
        entities=declared.get("entities", []),
        dimensions=dims if qtype == "comparison" else [],
        parts=declared.get("parts", slots if qtype == "synthesis" else []),
    )


def evaluate() -> dict[str, Any]:
    spec = json.loads((HERE / "cases.json").read_text())
    sources_by_id = {}
    results: list[dict[str, Any]] = []

    for case in spec["cases"]:
        sources_by_id = {s["id"]: s for s in case["sources"]}
        evidence_source = {e["id"]: e["source_id"] for e in case["evidence"]}
        for claim in case["claims"]:
            verdict = verify_claim(
                claim["text"],
                _cited(case, claim),
                _scorer(case, claim),
                support_threshold=THRESHOLD,
            )

            # What a generator would declare. An irrelevant claim does
            # not know it is irrelevant, so it optimistically claims
            # the core slot -- which is precisely the case the gate has
            # to refuse.
            contract = _contract(case)
            core = contract.core_slots[0].name if contract.core_slots else None
            declared_slot = _canonical(
                claim.get("declared_slot", claim.get("expected_slot") or core), contract
            )
            quotes = {e["id"]: e["quote"] for e in case["evidence"]}
            relevance = assess_relevance(
                claim["text"],
                declared_slot,
                contract,
                evidence_text=" ".join(quotes[e] for e in claim["cites"]),
                # The offline set measures the deterministic layer. A
                # permissive model is assumed so that anything rejected
                # here was rejected on structure, not on an opinion.
                model_says_relevant=True,
            )
            is_published = verdict.publishable and relevance.publishable
            chosen = verdict.best_evidence_id
            chosen_source = sources_by_id.get(evidence_source.get(chosen or "", ""), {})
            results.append(
                {
                    "case": case["id"],
                    "question_type": case["contract"]["question_type"],
                    "claim": claim["id"],
                    "published": is_published,
                    "supported": verdict.publishable,
                    "relevant": relevance.publishable,
                    "relevance_reason": relevance.reason,
                    "declared_slot": declared_slot,
                    "expected_publish": claim["expected_publish"],
                    "expected_relevant": claim["expected_relevant"],
                    "expected_slot": claim.get("expected_slot"),
                    "reason": verdict.reason,
                    "best_evidence": chosen,
                    "best_entailment": round(verdict.best_entailment, 4),
                    "selected_source_quality": chosen_source.get("quality"),
                    "selected_source_kind": chosen_source.get("kind"),
                    "expected_selected_evidence": claim.get("expected_selected_evidence"),
                }
            )

    published = [r for r in results if r["published"]]
    irrelevant_published = [r for r in published if not r["expected_relevant"]]
    should_have = [r for r in results if r["expected_publish"]]
    wrongly_withheld = [r for r in should_have if not r["published"]]
    selection_wrong = [
        r
        for r in results
        if r["expected_selected_evidence"] and r["best_evidence"] != r["expected_selected_evidence"]
    ]
    qualities = [r["selected_source_quality"] for r in published if r["selected_source_quality"]]

    summary = {
        "claims_evaluated": len(results),
        "published": len(published),
        "expected_published": len(should_have),
        "irrelevant_published": len(irrelevant_published),
        "irrelevant_publication_rate": round(len(irrelevant_published) / max(len(published), 1), 3),
        "correct_claims_wrongly_withheld": len(wrongly_withheld),
        "evidence_selection_wrong": len(selection_wrong),
        "primary_source_publications": sum(
            1 for r in published if r["selected_source_kind"] == "primary"
        ),
        "mean_selected_source_quality": round(sum(qualities) / len(qualities), 3)
        if qualities
        else None,
        "min_selected_source_quality": min(qualities) if qualities else None,
        "direct_answer_cases": 0,
        "zero_finding_cases": sum(
            1
            for case in spec["cases"]
            if not any(r["published"] for r in results if r["case"] == case["id"])
        ),
    }
    return {"summary": summary, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="write the full result here")
    args = parser.parse_args()

    report = evaluate()
    s = report["summary"]
    print("Adversarial quality evaluation")
    print(f"  claims evaluated              : {s['claims_evaluated']}")
    print(
        f"  published                     : {s['published']}  (expected {s['expected_published']})"
    )
    print(f"  IRRELEVANT published          : {s['irrelevant_published']}   <- must be 0")
    print(f"  correct claims wrongly withheld: {s['correct_claims_wrongly_withheld']}")
    print(f"  evidence selected wrongly     : {s['evidence_selection_wrong']}")
    print(
        f"  primary-source publications   : {s['primary_source_publications']} of {s['published']}"
    )
    print(f"  mean selected source quality  : {s['mean_selected_source_quality']}")
    print(f"  zero-finding cases            : {s['zero_finding_cases']}")
    print()
    for r in report["results"]:
        flag = " " if r["published"] == r["expected_publish"] else "!"
        got = "published" if r["published"] else "withheld "
        print(f" {flag} {r['case']:36} {r['claim']:3} {got}  {r['reason'][:58]}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
