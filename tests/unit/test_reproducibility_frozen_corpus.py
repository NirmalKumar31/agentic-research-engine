"""The frozen-corpus engine-determinism check, proven rather than assumed.

Three properties matter, and each gets its own test rather than one big
assertion: the run cannot reach the network under any circumstance, two
independent runs produce byte-identical manifests, and a broken engine
behaviour is caught rather than silently passing.

What this does *not* claim: nothing here says a real model or real search
would reproduce the same report. The frozen corpus scripts every external
boundary specifically so that question is out of scope -- see the module
docstring in ``evaluation/frozen_corpus.py``.
"""

from __future__ import annotations

import json
import socket

import pytest

from agentic_research.evaluation import frozen_corpus as fc


class TestTheRunCannotReachTheNetwork:
    async def test_a_real_socket_connection_would_be_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hard, low-level block, not merely "the fakes were used".

        `socket.create_connection` is what httpx's default transport (and
        anything else that opens a TCP connection) ultimately calls. If the
        frozen run ever reached a real network path -- a forgotten real
        client, a dependency that bypasses the scripted objects -- this
        fires before any byte leaves the process.
        """

        def refuse(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("frozen-corpus run attempted a real network connection")

        monkeypatch.setattr(socket, "create_connection", refuse)
        _state, manifest = await fc.run_frozen_corpus()
        assert manifest.ok, manifest.assertions

    async def test_the_scripted_objects_are_what_actually_ran(self) -> None:
        """Belt and braces alongside the socket block: the objects `RunContext`
        held were the scripted ones, not merely "nothing happened to need them"."""
        _state, manifest = await fc.run_frozen_corpus()
        assert manifest.assertions["search_and_fetch_are_the_scripted_objects"]


class TestDeterminism:
    async def test_two_independent_runs_match_byte_for_byte(self) -> None:
        _state_a, manifest_a = await fc.run_frozen_corpus()
        _state_b, manifest_b = await fc.run_frozen_corpus()
        a, b = manifest_a.to_dict(), manifest_b.to_dict()
        del a["git_commit"], b["git_commit"]  # only varies with uncommitted changes
        assert a == b

    async def test_the_report_hash_does_not_depend_on_wall_clock_time(self) -> None:
        """The regression this project actually hit.

        The rendered markdown ends with a generation timestamp -- correct
        for a real report, and it means hashing the *markdown* can never
        be reproducible across two runs made in different seconds. The
        manifest hashes structured report content instead. This test
        fixes the hash function's first argument in place across two
        calls a full second apart, which is what exposed the bug the
        first time this module was run for real.
        """
        import asyncio

        _state_a, manifest_a = await fc.run_frozen_corpus()
        await asyncio.sleep(1.1)
        _state_b, manifest_b = await fc.run_frozen_corpus()
        assert manifest_a.report_hash == manifest_b.report_hash

    def test_the_corpus_fingerprint_is_stable(self) -> None:
        assert fc.corpus_fingerprint() == fc.corpus_fingerprint()


class TestTheCorpusExercisesBothGates:
    """Non-vacuity: a corpus where everything predictably publishes would
    not prove the publication gate still refuses anything."""

    async def test_two_true_claims_publish(self) -> None:
        _state, manifest = await fc.run_frozen_corpus()
        assert manifest.assertions["supported_claim_published"]
        assert manifest.assertions["second_supported_claim_published"]

    async def test_one_overclaiming_claim_is_withheld(self) -> None:
        _state, manifest = await fc.run_frozen_corpus()
        assert manifest.assertions["unsupported_claim_withheld"]
        assert manifest.published_claim_count == 2
        assert manifest.withheld_claim_count == 1


class TestTheCommittedManifestMatchesTheCode:
    """This is the CI gate: a committed expected-manifest.json that must
    keep matching what the code in this branch actually produces.

    Written where the production path runs (`agentic-research
    verify-reproducible`), per this repo's own stated lesson about tests
    written where a value is declared instead of where it is used.
    """

    async def test_current_code_matches_the_committed_expectation(self) -> None:
        expected = fc.load_expected_manifest()
        assert expected is not None, (
            "no committed expected manifest -- generate one with "
            "`agentic-research verify-reproducible --write-expected`"
        )
        _state, manifest = await fc.run_frozen_corpus()
        problems = fc.diff_against_expected(manifest)
        assert not problems, problems

    def test_a_real_regression_is_not_silently_accepted(self) -> None:
        """Mutation check: corrupt the committed expectation and confirm
        the diff function actually reports it, rather than the comparison
        having quietly become a no-op."""

        real = fc.ReproducibilityManifest(
            corpus_fingerprint="deadbeef",
            engine_version="0.0.0",
            prompt_version="x",
            schema_version="y",
            git_commit="z",
            report_hash="mismatched",
            evidence_ids=("S1-e1",),
            published_claim_count=0,
            withheld_claim_count=0,
        )
        problems = fc.diff_against_expected(real)
        assert problems, "a manifest that disagrees with the committed one must report a problem"
        assert any("report_hash" in p for p in problems)

    def test_missing_expected_manifest_is_reported_not_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fc, "load_expected_manifest", lambda: None)
        manifest = fc.ReproducibilityManifest(
            corpus_fingerprint="a",
            engine_version="a",
            prompt_version="a",
            schema_version="a",
            git_commit="a",
            report_hash="a",
            evidence_ids=(),
            published_claim_count=0,
            withheld_claim_count=0,
        )
        problems = fc.diff_against_expected(manifest)
        assert problems == [
            "no expected manifest is committed yet -- run with --write-expected first"
        ]


class TestTheExpectedManifestFileItself:
    def test_it_is_valid_json_with_the_required_keys(self) -> None:
        path = fc.expected_manifest_path()
        assert path.is_file(), f"missing: {path}"
        data = json.loads(path.read_text())
        for key in (
            "corpus_fingerprint",
            "prompt_version",
            "schema_version",
            "report_hash",
            "evidence_ids",
            "assertions",
        ):
            assert key in data, key

    def test_its_own_assertions_are_all_true(self) -> None:
        """A committed expectation that bakes in a known-bad state would
        make every future check compare against a failure."""
        data = json.loads(fc.expected_manifest_path().read_text())
        assert all(data["assertions"].values()), data["assertions"]


class TestReplayReproducibility:
    """Level 1: replaying a committed recorded run makes no network call.

    This complements `test_replay_mode.py`'s payload/provenance/labelling
    assertions with the one thing they do not check: that replaying is
    not merely fast in practice, but structurally incapable of reaching
    the network, the same hard guarantee Level 2 proves for the engine
    itself.
    """

    def test_listing_recordings_touches_no_socket(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import socket

        from agentic_research.web import recordings

        def refuse(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("listing recordings attempted a real network connection")

        monkeypatch.setattr(socket, "create_connection", refuse)
        assert len(recordings.available()) >= 1

    def test_loading_and_rendering_a_recording_touches_no_socket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import socket

        from agentic_research.web import recordings

        def refuse(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("replaying a recording attempted a real network connection")

        monkeypatch.setattr(socket, "create_connection", refuse)
        summaries = recordings.available()
        for summary in summaries:
            payload = recordings.load(summary.id)
            assert payload.get("result", {}).get("markdown"), summary.id

    def test_every_recording_replays_through_the_cli(self) -> None:
        """The explicit command Level 1 asks for, exercised end to end."""
        from typer.testing import CliRunner

        from agentic_research.cli.main import app
        from agentic_research.web import recordings

        runner = CliRunner()
        for summary in recordings.available():
            result = runner.invoke(app, ["replay", summary.id])
            assert result.exit_code == 0, result.output

    def test_an_unknown_recording_id_fails_loudly_not_silently(self) -> None:
        from typer.testing import CliRunner

        from agentic_research.cli.main import app

        runner = CliRunner()
        result = runner.invoke(app, ["replay", "this-id-does-not-exist"])
        assert result.exit_code != 0
