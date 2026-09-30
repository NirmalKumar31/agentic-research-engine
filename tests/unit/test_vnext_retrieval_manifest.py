"""The next run must be diagnosable; the last one was not.

A hosted run on the causes of overfitting read six pages from 44 unique
candidates. Afterwards the other 38 were unrecoverable -- the service
runs with ``PERSIST_RUNS=false`` and the search stage streamed only
counts -- so "was there a better source in the pool that selection
passed over?" could not be answered, then or ever. Diagnosis had to
proceed from six survivors.

The manifest closes that for future runs. It is evaluation evidence and
it travels in the result payload, which the acceptance capture writes to
disk byte-for-byte, so it needs no server-side persistence and no
Render log access.

It also carries attacker-controlled text -- page titles and URLs come
from the open web -- so the redaction and injection tests here are not
hygiene. They are the reason a manifest can be written to a log, a JSON
artifact and a Markdown table without any of those being a new hole.
"""

from __future__ import annotations

import json

from agentic_research.evidence.dedup import Candidate
from agentic_research.models import DiscoveryRef, SearchQuery, SearchResult
from agentic_research.retrieval.manifest import (
    _MAX_CANDIDATES,
    build_manifest,
    safe_text,
    sanitize_url,
    with_fetch_outcomes,
)
from agentic_research.retrieval.selection import select_with_diagnostics


def candidate(
    url: str,
    *,
    score: float,
    sub_questions: tuple[str, ...] = ("SQ1",),
    title: str = "A page",
) -> Candidate:
    domain = url.split("/")[2]
    return Candidate(
        canonical_url=url,
        url=url,
        title=title,
        snippet="s",
        domain=domain,
        best_score=score,
        discovered_by=[
            DiscoveryRef(query_id=f"Q{i + 1}", sub_question_id=sq)
            for i, sq in enumerate(sub_questions)
        ],
    )


def query(qid: str, sq: str, text: str, rationale: str = "why") -> SearchQuery:
    return SearchQuery(id=qid, sub_question_id=sq, text=text, round_number=1, rationale=rationale)


def result(qid: str, url: str) -> SearchResult:
    return SearchResult(url=url, title="t", query_id=qid)


def manifest_for(pool: list[Candidate], *, limit: int = 2, order=("SQ1", "SQ2", "SQ3")):
    selection = select_with_diagnostics(
        pool, limit=limit, explanatory=True, sub_question_order=list(order)
    )
    return build_manifest(
        round_number=1,
        queries=[query("Q1", "SQ1", "overfitting causes machine learning")],
        results=[result("Q1", c.url) for c in pool],
        candidates=pool,
        selection=selection,
        explanatory=True,
    )


class TestEveryCandidateIsAccountedForExactlyOnce:
    def test_selected_and_dropped_partition_the_pool(self) -> None:
        pool = [
            candidate("https://arxiv.org/abs/1", score=0.90),
            candidate("https://x.com/a/status/1", score=0.92),
            candidate("https://codefinity.com/p", score=0.70, sub_questions=("SQ2",)),
            candidate("https://medium.com/p", score=0.60, sub_questions=("SQ3",)),
        ]
        manifest = manifest_for(pool)
        assert len(manifest.candidates) == len(pool)
        assert len(manifest.selected) + len(manifest.dropped) == len(pool)
        urls = [c.url for c in manifest.candidates]
        assert len(urls) == len(set(urls)), "a candidate appears twice"

    def test_the_selected_set_matches_what_was_chosen(self) -> None:
        pool = [
            candidate("https://arxiv.org/abs/1", score=0.90),
            candidate("https://x.com/a/status/1", score=0.92),
        ]
        selection = select_with_diagnostics(
            pool, limit=1, explanatory=True, sub_question_order=["SQ1"]
        )
        manifest = build_manifest(
            round_number=1,
            queries=[query("Q1", "SQ1", "q")],
            results=[result("Q1", c.url) for c in pool],
            candidates=pool,
            selection=selection,
            explanatory=True,
        )
        chosen = {sanitize_url(c.url) for c in selection.selected}
        assert {c.url for c in manifest.selected} == chosen

    def test_every_dropped_candidate_carries_a_reason(self) -> None:
        pool = [
            candidate("https://arxiv.org/abs/1", score=0.90),
            candidate("https://x.com/a/status/1", score=0.92),
            candidate("https://medium.com/p", score=0.50),
        ]
        manifest = manifest_for(pool, limit=1, order=("SQ1",))
        assert manifest.dropped
        for record in manifest.dropped:
            assert record.reason and record.reason != "not selected", record

    def test_a_deprioritised_drop_is_distinguishable_from_a_budget_drop(self) -> None:
        """Two causes, two fixes. They were indistinguishable before."""
        pool = [
            candidate("https://arxiv.org/abs/1", score=0.90),
            candidate("https://x.com/a/status/1", score=0.92),
        ]
        manifest = manifest_for(pool, limit=1, order=("SQ1",))
        reason = next(c.reason for c in manifest.dropped if c.domain == "x.com")
        assert "social" in reason or "deprioritised" in reason


class TestMultiSubQuestionAssociationsSurvive:
    def test_every_association_is_recorded(self) -> None:
        pool = [
            candidate("https://arxiv.org/abs/1", score=0.9, sub_questions=("SQ1", "SQ2", "SQ3"))
        ]
        manifest = manifest_for(pool, limit=1)
        record = manifest.candidates[0]
        assert record.sub_question_ids == ("SQ1", "SQ2", "SQ3")

    def test_the_allocation_map_credits_all_of_them(self) -> None:
        pool = [candidate("https://arxiv.org/abs/1", score=0.9, sub_questions=("SQ1", "SQ2"))]
        manifest = manifest_for(pool, limit=1)
        assert manifest.by_sub_question["SQ1"] == manifest.by_sub_question["SQ2"]
        assert manifest.by_sub_question["SQ1"]


class TestTheOrderingCanBeRecomputed:
    def test_the_scores_reproduce_the_selected_ordering(self) -> None:
        """A single total cannot be audited. Provider relevance and the
        adjustment are recorded separately so a reader can check whether
        a page ranked highly for being relevant or for being
        accountable."""
        pool = [
            candidate("https://x.com/a/status/1", score=0.92),
            candidate("https://arxiv.org/abs/1", score=0.90),
            candidate("https://medium.com/p", score=0.85),
        ]
        manifest = manifest_for(pool, limit=1, order=("SQ1",))
        for record in manifest.candidates:
            assert record.ranking_score == (record.provider_score + record.authority_adjustment)
        best = max(manifest.candidates, key=lambda c: c.ranking_score)
        assert best.selected, "the top-scoring candidate was not the one selected"

    def test_the_adjustment_shows_its_sign(self) -> None:
        pool = [
            candidate("https://arxiv.org/abs/1", score=0.80),
            candidate("https://x.com/a/status/1", score=0.80),
        ]
        manifest = manifest_for(pool, limit=1, order=("SQ1",))
        by_domain = {c.domain: c for c in manifest.candidates}
        assert by_domain["arxiv.org"].authority_adjustment > 0
        assert by_domain["x.com"].authority_adjustment < 0

    def test_queries_record_their_purpose_and_yield(self) -> None:
        pool = [candidate("https://arxiv.org/abs/1", score=0.9)]
        manifest = manifest_for(pool, limit=1)
        record = manifest.queries[0]
        assert record.query_id == "Q1"
        assert record.sub_question_id == "SQ1"
        assert record.rationale == "why"
        assert record.result_count == 1

    def test_a_query_that_returned_nothing_is_still_listed(self) -> None:
        """Counted rather than tracked, so a zero-yield query appears as
        `results: 0` instead of being absent -- which is the difference
        between a bad query and a query nobody sent."""
        selection = select_with_diagnostics([], limit=1, sub_question_order=["SQ1"])
        manifest = build_manifest(
            round_number=1,
            queries=[query("Q1", "SQ1", "a query with no hits")],
            results=[],
            candidates=[],
            selection=selection,
            explanatory=True,
        )
        assert manifest.queries[0].result_count == 0


class TestRedaction:
    """Everything stripped here is somewhere a secret has been found."""

    def test_credentials_in_userinfo_are_removed(self) -> None:
        assert sanitize_url("https://user:s3cret@example.com/a") == "https://example.com/a"

    def test_query_strings_are_removed(self) -> None:
        for url in [
            "https://example.com/a?api_key=sk-live-abcdef",
            "https://example.com/a?token=eyJhbGciOi&x=1",
            "https://example.com/a?utm_source=x&sig=deadbeef",
        ]:
            out = sanitize_url(url)
            assert "?" not in out
            assert out == "https://example.com/a"

    def test_fragments_are_removed(self) -> None:
        assert sanitize_url("https://example.com/a#access_token=xyz") == "https://example.com/a"

    def test_the_port_is_kept_because_it_is_not_sensitive(self) -> None:
        assert sanitize_url("https://example.com:8443/a") == "https://example.com:8443/a"

    def test_a_malformed_url_yields_nothing_rather_than_itself(self) -> None:
        for bad in ["not a url", "", "javascript:alert(1)", "//example.com/a"]:
            assert sanitize_url(bad) in ("", "javascript:")

    def test_no_secret_survives_into_the_serialised_manifest(self) -> None:
        pool = [
            candidate(
                "https://user:pw@example.com/p?api_key=sk-live-SECRET&session=abc#tok=XYZ",
                score=0.9,
                title="Totally normal page",
            )
        ]
        manifest = manifest_for(pool, limit=1)
        payload = json.dumps(manifest.to_dict())
        for secret in ["sk-live-SECRET", "session=abc", "tok=XYZ", "user:pw", "api_key"]:
            assert secret not in payload, secret
        # The allocation map was the leak: it holds raw canonical URLs
        # from the selection result, and copying it verbatim carried
        # the query string through while every candidate record was
        # clean. Asserted separately so a future copy cannot reopen it.
        for urls in manifest.by_sub_question.values():
            for url in urls:
                assert "?" not in url and "@" not in url and "#" not in url

    def test_the_manifest_carries_no_page_body_or_raw_payload(self) -> None:
        pool = [candidate("https://example.com/p", score=0.9)]
        record = manifest_for(pool, limit=1).candidates[0]
        fields = set(record.to_dict())
        for forbidden in ("raw_content", "body", "snippet", "headers", "cookies", "text"):
            assert forbidden not in fields


class TestInjection:
    """Titles and URLs are attacker-controlled text from the open web."""

    def test_newlines_cannot_forge_a_log_line(self) -> None:
        assert "\n" not in safe_text("Title\nlevel=error msg=forged")
        assert "\r" not in safe_text("Title\r\nforged")

    def test_markdown_table_cells_cannot_be_split(self) -> None:
        assert "|" not in safe_text("Title | injected cell")

    def test_markup_and_backticks_are_defanged(self) -> None:
        out = safe_text("<script>alert(1)</script> `code`")
        assert "<" not in out and ">" not in out and "`" not in out

    def test_a_hostile_title_is_neutralised_in_the_manifest(self) -> None:
        pool = [
            candidate(
                "https://example.com/p",
                score=0.9,
                title="ok\nlevel=error | <img src=x onerror=alert(1)>",
            )
        ]
        payload = json.dumps(manifest_for(pool, limit=1).to_dict())
        assert "\\n" not in payload
        assert "<img" not in payload

    def test_text_is_truncated_visibly(self) -> None:
        out = safe_text("x" * 5000)
        assert len(out) <= 300
        assert out.endswith("…")


class TestBoundedness:
    def test_the_candidate_list_is_capped_and_says_so(self) -> None:
        pool = [
            candidate(f"https://e{i}.example.com/p", score=0.5, sub_questions=("SQ1",))
            for i in range(_MAX_CANDIDATES + 25)
        ]
        manifest = manifest_for(pool, limit=2, order=("SQ1",))
        assert len(manifest.candidates) == _MAX_CANDIDATES
        assert manifest.truncated == 25, "a silent cut would look complete"

    def test_a_normal_round_is_nowhere_near_the_cap(self) -> None:
        """Non-vacuity for the cap: it guards against a provider
        returning more than it advertises, not a routine round."""
        pool = [candidate(f"https://e{i}.example.com/p", score=0.5) for i in range(48)]
        manifest = manifest_for(pool, limit=6, order=("SQ1",))
        assert manifest.truncated == 0
        assert len(manifest.candidates) == 48


class TestFetchOutcomesAreMergedIn:
    def _selected_manifest(self) -> dict:
        pool = [candidate("https://example.com/p", score=0.9)]
        return manifest_for(pool, limit=1).to_dict()

    def test_a_fetched_source_records_its_status_and_origin(self) -> None:
        from agentic_research.models import ContentOrigin, FetchStatus, SourceDocument

        source = SourceDocument(
            id="S1",
            url="https://example.com/p",
            canonical_url="https://example.com/p",
            title="t",
            domain="example.com",
            fetch_status=FetchStatus.OK,
            content_origin=ContentOrigin.HTML_FETCH,
        )
        merged = with_fetch_outcomes(self._selected_manifest(), [source])
        outcome = merged["candidates"][0]["fetch_outcome"]
        assert outcome.startswith("ok")
        assert "html_fetch" in outcome

    def test_an_http_failure_is_distinguishable_from_not_being_tried(self) -> None:
        from agentic_research.models import FetchStatus, SourceDocument

        failed = SourceDocument(
            id="S1",
            url="https://example.com/p",
            canonical_url="https://example.com/p",
            title="t",
            domain="example.com",
            fetch_status=FetchStatus.HTTP_ERROR,
        )
        merged = with_fetch_outcomes(self._selected_manifest(), [failed])
        assert "http_error" in merged["candidates"][0]["fetch_outcome"]

        untried = with_fetch_outcomes(self._selected_manifest(), [])
        assert untried["candidates"][0]["fetch_outcome"] == "not fetched"

    def test_a_dropped_candidate_is_not_given_a_fetch_outcome(self) -> None:
        pool = [
            candidate("https://arxiv.org/abs/1", score=0.9),
            candidate("https://x.com/a/status/1", score=0.95),
        ]
        merged = with_fetch_outcomes(manifest_for(pool, limit=1, order=("SQ1",)).to_dict(), [])
        dropped = [c for c in merged["candidates"] if not c["selected"]]
        assert dropped
        assert all(c["fetch_outcome"] == "not fetched" for c in dropped)


class TestItReachesTheArtifact:
    def test_the_node_records_it_in_state(self) -> None:
        import inspect

        from agentic_research.graph.nodes import research

        body = inspect.getsource(research.dedupe_sources)
        assert '"retrieval_manifest"' in body
        assert "build_manifest(" in body

    def test_the_result_payload_carries_it_with_fetch_outcomes(self) -> None:
        import inspect

        from agentic_research.web import recordings

        body = inspect.getsource(recordings)
        assert '"retrieval_manifest"' in body
        assert "with_fetch_outcomes(" in body

    def test_older_recordings_backfill_to_an_empty_list(self) -> None:
        from agentic_research.web.recordings import _backfill_additive_keys

        payload: dict = {"result": {}}
        _backfill_additive_keys(payload)
        assert payload["result"]["retrieval_manifest"] == []
