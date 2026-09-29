"""Committed run evidence stays verifiable, sanitised, and honest.

Numbers in a README are worth what a reader can recompute. Two runs are
recorded: one from the CLI under local budgets, one through the
deployed endpoint under the public ones. They have different shapes and
different claims attached, so the invariants below split into the ones
every artifact must satisfy and the ones specific to each kind.

The reconciliation test exists because a summary of the first run got
the withheld total wrong by counting only 'unsupported' and forgetting
that partially supported claims are withheld too.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "examples" / "live-validation"

# Every artifact carries these.
CORE = {
    "README.md",
    "checksums.sha256",
    "environment.json",
    "manual-review.md",
    "metrics.json",
    "report.md",
    "sources.json",
}
# A hosted run additionally carries its raw capture.
HOSTED_EXTRA = {"stream.raw.sse", "stream.index.jsonl", "withheld-reasons.json"}


# Directories under examples/live-validation that are not runs.
# `failures` holds captures of runs that produced no result: they have
# no report and no metrics, so the contract below cannot apply, and a
# directory pretending otherwise is worse than none.
#
# Named explicitly rather than inferred from their contents: a rule like
# "has metrics.json" would make a run that is missing its metrics
# vanish from the parametrisation instead of failing, which is the one
# thing this file exists to catch.
NOT_A_RUN = {"tools", "failures"}


def runs() -> list[Path]:
    if not VALIDATION.is_dir():
        return []
    return sorted(p for p in VALIDATION.iterdir() if p.is_dir() and p.name not in NOT_A_RUN)


def is_hosted(run_dir: Path) -> bool:
    return (run_dir / "stream.raw.sse").is_file()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "run_dir" in metafunc.fixturenames:
        found = runs()
        metafunc.parametrize("run_dir", found, ids=[p.name for p in found] or None)


class TestTheArtifactIsComplete:
    def test_at_least_one_run_is_committed(self) -> None:
        assert runs()

    def test_the_exclusion_list_hides_no_real_run(self) -> None:
        """A name added to NOT_A_RUN removes a directory from every
        check in this file. It must never be able to do that to a run."""
        for name in NOT_A_RUN:
            directory = VALIDATION / name
            if not directory.is_dir():
                continue
            assert not (directory / "metrics.json").exists(), (
                f"{name} looks like a run but is excluded from every assertion"
            )

    def test_a_hosted_run_is_committed(self) -> None:
        """Deployment acceptance needs evidence from the deployment."""
        assert any(is_hosted(r) for r in runs())

    def test_every_required_file_is_present(self, run_dir: Path) -> None:
        present = {p.name for p in run_dir.iterdir()}
        required = CORE | (HOSTED_EXTRA if is_hosted(run_dir) else set())
        assert required <= present, f"missing: {required - present}"

    def test_checksums_match(self, run_dir: Path) -> None:
        """Catches a file edited without regenerating the manifest,
        which is how an artifact quietly stops describing its run."""
        listed = 0
        for line in (run_dir / "checksums.sha256").read_text().splitlines():
            if not line.strip():
                continue
            digest, name = line.split(maxsplit=1)
            target = run_dir / name.strip()
            assert target.exists(), f"{name} is in the manifest but missing"
            assert hashlib.sha256(target.read_bytes()).hexdigest() == digest, (
                f"{name} does not match its recorded checksum"
            )
            listed += 1
        assert listed >= len(CORE) - 1  # the manifest cannot list itself


class TestTheCountsReconcile:
    def test_every_checked_claim_lands_in_exactly_one_category(self, run_dir: Path) -> None:
        rec = json.loads((run_dir / "metrics.json").read_text())["claims_reconciliation"]
        total = rec.get("generated_and_checked", rec.get("checked"))
        assert rec["published"] + rec["withheld"] + rec["not_checked"] == total

    def test_partially_supported_counts_as_withheld(self, run_dir: Path) -> None:
        """The specific mistake this file exists to prevent."""
        metrics = json.loads((run_dir / "metrics.json").read_text())
        rec = metrics["claims_reconciliation"]
        if rec["withheld_partially_supported"]:
            assert rec["withheld"] > rec["withheld_unsupported"]

    def test_published_never_exceeds_checked(self, run_dir: Path) -> None:
        rec = json.loads((run_dir / "metrics.json").read_text())["claims_reconciliation"]
        total = rec.get("generated_and_checked", rec.get("checked"))
        assert 0 <= rec["published"] <= total

    def test_quote_fidelity_denominator_is_defined(self, run_dir: Path) -> None:
        metrics = json.loads((run_dir / "metrics.json").read_text())
        fidelity = metrics["quote_fidelity"]
        assert fidelity["denominator_extracted_quotes"] == (
            metrics["exact_quotes"] + metrics["fuzzy_quotes"] + metrics["unmatched_quotes"]
        )
        assert "exact normalised substring" in fidelity["definition"].lower()


class TestCliArtifactSpecifics:
    def test_published_claims_are_listed_with_their_quotes(self, run_dir: Path) -> None:
        if is_hosted(run_dir):
            pytest.skip("hosted artifacts carry the raw stream instead")
        published = json.loads((run_dir / "published-claims.json").read_text())
        rec = json.loads((run_dir / "metrics.json").read_text())["claims_reconciliation"]
        assert len(published) == rec["published"]
        for entry in published:
            assert entry["evidence"]
            for ev in entry["evidence"]:
                assert ev["quote"].strip()
                assert ev["source_url"]

    def test_it_does_not_call_itself_hosted_acceptance(self, run_dir: Path) -> None:
        if is_hosted(run_dir):
            pytest.skip("this one is hosted acceptance")
        env = json.loads((run_dir / "environment.json").read_text())
        assert env["not_a_hosted_acceptance"] is True
        assert env["executed_by"] == "local CLI"


class TestHostedArtifactSpecifics:
    def test_it_records_the_deployed_commit(self, run_dir: Path) -> None:
        if not is_hosted(run_dir):
            pytest.skip("CLI artifact")
        env = json.loads((run_dir / "environment.json").read_text())
        assert re.fullmatch(r"[0-9a-f]{7,40}", env["deployed_commit"])
        assert env["is_hosted_acceptance"] is True

    def test_it_does_not_claim_to_evaluate_quality(self, run_dir: Path) -> None:
        """One run is not a benchmark, and a zero-publication run is not
        a quality result in either direction."""
        if not is_hosted(run_dir):
            pytest.skip("CLI artifact")
        env = json.loads((run_dir / "environment.json").read_text())
        assert env["is_research_quality_evaluation"] is False

    def test_the_capture_is_complete(self, run_dir: Path) -> None:
        """The previous capture truncated its lines and destroyed the
        result payload, which the service does not persist."""
        if not is_hosted(run_dir):
            pytest.skip("CLI artifact")
        raw = (run_dir / "stream.raw.sse").read_text()
        assert "event: result" in raw
        assert "event: done" in raw
        payload = re.search(r"^event: result\ndata: (.*)$", raw, re.M)
        assert payload, "no result payload in the capture"
        parsed = json.loads(payload.group(1))
        assert {"report", "metrics", "verification", "sources"} <= set(parsed)

    def test_the_capture_matches_the_derived_counts(self, run_dir: Path) -> None:
        """Derived metrics must come from the committed stream, not from
        a summary written alongside it."""
        if not is_hosted(run_dir):
            pytest.skip("CLI artifact")
        raw = (run_dir / "stream.raw.sse").read_text()
        parsed = json.loads(re.search(r"^event: result\ndata: (.*)$", raw, re.M).group(1))
        recorded = json.loads((run_dir / "metrics.json").read_text())
        assert recorded["support_breakdown"] == parsed["metrics"]["support_breakdown"]
        assert recorded["known_cost_usd"] == parsed["metrics"]["known_cost_usd"]

    def test_a_zero_publication_run_says_so(self, run_dir: Path) -> None:
        if not is_hosted(run_dir):
            pytest.skip("CLI artifact")
        rec = json.loads((run_dir / "metrics.json").read_text())["claims_reconciliation"]
        if rec["published"] == 0:
            review = (run_dir / "manual-review.md").read_text().lower()
            assert "nothing was published" in review
            readme = (run_dir / "README.md").read_text().lower()
            assert "replay" in readme, (
                "a zero-publication run must say what the better demonstration is"
            )


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
    def test_the_cost_figure_is_scoped(self, run_dir: Path) -> None:
        env = json.loads((run_dir / "environment.json").read_text())
        cost = env["cost"]
        assert any("openai" in k for k in cost)
        blob = json.dumps(cost).lower()
        assert "not measured" in blob or "not separately measured" in blob
        assert cost.get("total_cross_provider_cost_complete") is False or (
            "not stated" in str(cost.get("total", "")).lower()
        )

    def test_completeness_is_stated_per_provider_not_overall(self, run_dir: Path) -> None:
        metrics = json.loads((run_dir / "metrics.json").read_text())
        definitions = metrics.get("cost_field_definitions", {})
        assert definitions, "cost fields are undefined"
        assert "total_cross_provider_cost_complete" in json.dumps(metrics)

    def test_the_readme_gives_the_verification_command(self, run_dir: Path) -> None:
        readme = (run_dir / "README.md").read_text()
        assert "shasum -a 256 -c checksums.sha256" in readme
        assert run_dir.name in readme

    def test_source_quality_is_not_called_a_truth_score(self, run_dir: Path) -> None:
        readme = (run_dir / "README.md").read_text().lower()
        if "quality 0." in readme or "0.95" in readme:
            assert "heuristic" in readme
