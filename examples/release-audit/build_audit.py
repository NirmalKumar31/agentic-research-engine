"""Build the candidate-level release audit from the canonical recordings.

Every atomic candidate the synthesiser produced, with its evidence, its
guard results, its pairwise NLI scores and whether it reached the
published report. Written from the recordings themselves rather than
re-run, so the artifact and the demo the site replays are the same run.

The release metric this exists to support is one number: how many
published claims a human reader judges unsupported by their own cited
evidence. That judgement is not automatable and is not attempted here;
this produces the reviewable list and the counts around it.

Run:  python examples/release-audit/build_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agentic_research.citations.atomicity import compound_markers  # noqa: E402

RECORDINGS = ROOT / "src" / "agentic_research" / "web" / "recorded_runs"
HERE = Path(__file__).parent


def _published_texts(result: dict) -> set[str]:
    """Claim text as it appears in the report the reader is shown."""
    report = result.get("report") or {}
    texts = {c["text"] for c in report.get("summary_claims", [])}
    texts |= {c["text"] for c in report.get("key_findings", [])}
    for section in report.get("sections", []):
        texts |= {c["text"] for c in section.get("claims", [])}
    return texts


def main() -> None:
    audit: dict[str, object] = {
        "description": (
            "Candidate-level audit of the three canonical runs. One record per unique "
            "post-deduplication substantive candidate, with the complete claim text, its "
            "cited evidence, every pairwise NLI score, every guard result, and whether it "
            "reached the published report. Immutable: regenerate only by re-recording."
        ),
        "publication_rule": (
            "A claim publishes when at least one cited, citable quote passes every "
            "deterministic guard and entails the claim at or above the support threshold. "
            "Quotes are scored individually and never concatenated."
        ),
        "runs": {},
    }
    totals = {"candidates": 0, "checked": 0, "published": 0, "withheld": 0, "excerpts": 0}

    for path in sorted(RECORDINGS.glob("*.json")):
        recording = json.loads(path.read_text())
        meta, result = recording["meta"], recording["result"]
        verification = result.get("verification") or {}
        judgments = verification.get("judgments") or []
        published = _published_texts(result)

        evidence = {e["id"]: e for e in result.get("evidence", [])}
        records = []
        for judgment in judgments:
            cited = []
            for evidence_id in judgment.get("evidence_ids", []):
                item = evidence.get(evidence_id, {})
                cited.append(
                    {
                        "evidence_id": evidence_id,
                        "quote": item.get("quote"),
                        "source_id": item.get("source_id"),
                        "page": item.get("page"),
                        "quote_match": item.get("quote_match"),
                    }
                )
            in_report = judgment["claim_text"] in published
            records.append(
                {
                    "claim": judgment["claim_text"],
                    "kind": judgment.get("kind"),
                    "evidence": cited,
                    "nli_scores": judgment.get("evidence_scores", []),
                    "best_evidence_id": judgment.get("best_evidence_id"),
                    "best_entailment": judgment.get("best_entailment"),
                    "model_id": judgment.get("model_id"),
                    "model_revision": judgment.get("model_revision"),
                    "support_threshold": judgment.get("support_threshold"),
                    "checked": judgment.get("checked", False),
                    "publishable": judgment.get("publishable", False),
                    "diagnostic_verdict": judgment.get("verdict"),
                    "withhold_reason": judgment.get("reason"),
                    "present_in_published_report": in_report,
                    "compound_markers": compound_markers(judgment["claim_text"]),
                    # Filled in by a human. The release gate is this
                    # column, and nothing computes it.
                    "human_review": None,
                }
            )

        # The gate and the rendered report must agree. A claim marked
        # unpublishable that still appears is the failure this audit is
        # for, so it is surfaced rather than counted.
        leaked = [
            r["claim"] for r in records if r["present_in_published_report"] and not r["publishable"]
        ]

        counts = {
            "raw_generated": verification.get("generated_substantive_claims", 0),
            "duplicates_removed": verification.get("duplicate_claims_removed", 0),
            "candidates": len(records),
            "checkable": verification.get("checkable_claims", 0),
            "checked": verification.get("checked_claims", 0),
            "not_checked": verification.get("not_checked_claims", 0),
            "published": verification.get("final_published_claims", 0),
            "withheld": sum(1 for r in records if not r["publishable"]),
            "evidence_only_excerpts": verification.get("evidence_only_excerpts", 0),
            "exhaustive": verification.get("entailment_exhaustive", False),
        }
        checked = counts["checked"]
        counts["publication_rate"] = round(counts["published"] / checked, 4) if checked else 0.0

        audit["runs"][meta["id"]] = {  # type: ignore[index]
            "question": meta["question"],
            "recorded_at": meta.get("recorded_at"),
            "commit": meta.get("commit"),
            "counts": counts,
            "gate_disagreements": leaked,
            "candidates": records,
        }
        for key in ("candidates", "checked", "published", "withheld"):
            totals[key] += counts[key if key != "candidates" else "candidates"]
        totals["excerpts"] += counts["evidence_only_excerpts"]

    audit["totals"] = totals
    out = HERE / "candidate-audit.json"
    out.write_text(json.dumps(audit, indent=2) + "\n")

    print(f"{'run':<32} {'cand':>5} {'chk':>5} {'pub':>5} {'wh':>5} {'exc':>5}  rate")
    for name, run in audit["runs"].items():  # type: ignore[attr-defined]
        c = run["counts"]
        print(
            f"{name:<32} {c['candidates']:>5} {c['checked']:>5} {c['published']:>5} "
            f"{c['withheld']:>5} {c['evidence_only_excerpts']:>5}  {c['publication_rate']:.0%}"
        )
        if run["gate_disagreements"]:
            print(f"  GATE DISAGREEMENT: {run['gate_disagreements']}")
    print(f"\ntotals: {totals}")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
