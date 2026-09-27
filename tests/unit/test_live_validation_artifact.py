"""The committed live-run evidence stays verifiable and stays sanitised.

Numbers quoted in a README are worth what the reader can check. This
run's artifact exists so a reviewer can recompute the counts instead of
believing a summary -- one of which already got the withheld total
wrong by counting only 'unsupported' and forgetting that partially
supported claims are withheld too.

So the arithmetic is asserted rather than described, and so is the
absence of anything that should not have been committed.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "examples" / "live-validation"

REQUIRED = {
    "README.md",
    "checksums.sha256",
    "environment.json",
    "manual-review.md",
    "metrics.json",
    "published-claims.json",
    "report.md",
    "sources.json",
}


def runs() -> list[Path]:
    return sorted(p for p in VALIDATION.iterdir() if p.is_dir()) if VALIDATION.is_dir() else []


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "run_dir" in metafunc.fixturenames:
        found = runs()
        metafunc.parametrize("run_dir", found, ids=[p.name for p in found] or None)


class TestTheArtifactIsComplete:
    def test_at_least_one_run_is_committed(self) -> None:
        assert runs(), "no live-validation run is committed"

    def test_every_required_file_is_present(self, run_dir: Path) -> None:
        assert {p.name for p in run_dir.iterdir()} >= REQUIRED

    def test_checksums_match(self, run_dir: Path) -> None:
        """Catches a file edited without regenerating the manifest,
        which is how a committed artifact quietly stops describing the
        run it came from."""
        listed = 0
        for line in (run_dir / "checksums.sha256").read_text().splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            target = run_dir / name.strip()
            assert target.exists(), f"{name} is in the manifest but missing"
            actual = hashlib.sha256(target.read_bytes()).hexdigest()
            assert actual == digest, f"{name} does not match its recorded checksum"
            listed += 1
        assert listed >= len(REQUIRED) - 1  # the manifest cannot list itself


class TestTheCountsReconcile:
    def test_every_checked_claim_lands_in_exactly_one_category(self, run_dir: Path) -> None:
        """generated = published + withheld + not-checked, with nothing
        double-counted and nothing unaccounted for."""
        metrics = json.loads((run_dir / "metrics.json").read_text())
        breakdown = metrics["support_breakdown"]
        rec = metrics["claims_reconciliation"]

        assert rec["generated_and_checked"] == sum(breakdown.values())
        assert rec["published"] == breakdown["supported"]
        assert rec["withheld"] == (breakdown["partially_supported"] + breakdown["unsupported"])
        assert rec["not_checked"] == breakdown["not_checked"]
        assert (
            rec["published"] + rec["withheld"] + rec["not_checked"] == rec["generated_and_checked"]
        )

    def test_partially_supported_counts_as_withheld(self, run_dir: Path) -> None:
        """The specific mistake this file exists to prevent. Only
        'supported' publishes."""
        metrics = json.loads((run_dir / "metrics.json").read_text())
        breakdown = metrics["support_breakdown"]
        rec = metrics["claims_reconciliation"]
        if breakdown["partially_supported"]:
            assert rec["withheld"] > breakdown["unsupported"], (
                "withheld equals the unsupported count, so partially "
                "supported claims were left out of it"
            )

    def test_published_claims_match_the_published_count(self, run_dir: Path) -> None:
        published = json.loads((run_dir / "published-claims.json").read_text())
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert len(published) == metrics["claims_reconciliation"]["published"]

    def test_every_published_claim_cites_evidence_with_a_quote(self, run_dir: Path) -> None:
        for entry in json.loads((run_dir / "published-claims.json").read_text()):
            assert entry["evidence"], f"published with no evidence: {entry['claim'][:60]}"
            for ev in entry["evidence"]:
                assert ev["quote"].strip(), "cited evidence carries no quote"
                assert ev["source_url"], "cited evidence resolves to no source"

    def test_quote_fidelity_denominator_is_defined(self, run_dir: Path) -> None:
        metrics = json.loads((run_dir / "metrics.json").read_text())
        fidelity = metrics["quote_fidelity"]
        assert fidelity["denominator_extracted_quotes"] == (
            metrics["exact_quotes"] + metrics["fuzzy_quotes"] + metrics["unmatched_quotes"]
        )
        assert "exact normalised substring" in fidelity["definition"].lower()


class TestNothingSensitiveWasCommitted:
    FORBIDDEN = {
        "hugging face endpoint url": re.compile(r"[a-z0-9]{16,}\.endpoints\.huggingface\.cloud"),
        "local path": re.compile(r"(/Users/|/home/[a-z])"),
        "bearer token": re.compile(r"Bearer\s+[A-Za-z0-9_\-.]{12,}"),
        "hf token": re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
        "openai key": re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
        "tavily key": re.compile(r"\btvly-[A-Za-z0-9]{20,}"),
        "redis url": re.compile(r"redis://\S*:\S*@"),
    }

    def test_no_credential_or_private_location_appears(self, run_dir: Path) -> None:
        offenders: list[str] = []
        for path in sorted(run_dir.iterdir()):
            text = path.read_text(errors="ignore")
            for label, pattern in self.FORBIDDEN.items():
                if pattern.search(text):
                    offenders.append(f"{path.name}: {label}")
        assert not offenders, f"sensitive content committed: {offenders}"


class TestTheClaimsAreHonest:
    def test_it_does_not_call_itself_hosted_acceptance(self, run_dir: Path) -> None:
        env = json.loads((run_dir / "environment.json").read_text())
        assert env["not_a_hosted_acceptance"] is True
        assert env["executed_by"] == "local CLI"

    def test_it_records_that_budgets_differed_from_the_public_ones(self, run_dir: Path) -> None:
        """A run under looser limits than the deployment cannot stand in
        for one under the deployment's limits."""
        env = json.loads((run_dir / "environment.json").read_text())
        public = env["public_deployment_limits_for_comparison"]
        effective = env["effective_budgets"]
        assert effective["max_sources"] != public["max_sources"]
        assert "note" in public

    def test_the_cost_figure_is_scoped_to_one_provider(self, run_dir: Path) -> None:
        env = json.loads((run_dir / "environment.json").read_text())
        cost = env["cost"]
        assert "openai_engine_calculated_usd" in cost
        assert "not measured" in str(cost["tavily_usd"]).lower()
        assert "not separately measured" in str(cost["huggingface_endpoint_usd"]).lower()
        assert "not stated" in str(cost["total"]).lower()

    def test_the_engine_commit_is_recorded_and_not_assumed(self, run_dir: Path) -> None:
        env = json.loads((run_dir / "environment.json").read_text())
        assert re.fullmatch(r"[0-9a-f]{40}", env["engine_commit"])
        assert env["engine_commit_note"]

    def test_the_manual_review_covers_every_published_claim(self, run_dir: Path) -> None:
        published = json.loads((run_dir / "published-claims.json").read_text())
        review = (run_dir / "manual-review.md").read_text()
        verdicts = len(re.findall(r"^## \d+\. (PASS|FAIL)", review, re.M))
        assert verdicts == len(published), (
            f"{verdicts} reviewed findings for {len(published)} published claims"
        )

    def test_source_quality_is_called_heuristic(self, run_dir: Path) -> None:
        readme = (run_dir / "README.md").read_text().lower()
        assert "heuristic" in readme
