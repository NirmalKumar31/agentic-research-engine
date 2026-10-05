"""`scripts/check_docs.py` is the gate that stops a documentation
regression from reaching CI undetected -- so the gate itself needs a
regression test, not just a one-off manual smoke test.

The most important case here is `test_check_links_rejects_untracked_target`:
an earlier version of this checker resolved links against local disk
(`Path.exists()`), which let a link to a file that exists on the
machine running the check -- but is not committed -- pass locally and
then break on CI's fresh checkout. That exact bug reached a pushed
commit once (`evaluations/phase_b/README.md` linking the intentionally
uncommitted `adjudication/README.md`) before CI caught it. Every test
below that builds an isolated git repo exists to make sure that class
of bug fails locally, every time, before a push.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import check_docs  # noqa: E402


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    return repo


def _write(repo: Path, rel: str, text: str) -> Path:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


class TestTrackedFiles:
    def test_untracked_file_is_not_in_the_tracked_set(self, git_repo: Path) -> None:
        _write(git_repo, "committed.md", "# committed\n")
        _git(git_repo, "add", "committed.md")
        _write(git_repo, "scratch.md", "# scratch, never added\n")

        tracked = check_docs.tracked_files(git_repo)

        assert "committed.md" in tracked
        assert "scratch.md" not in tracked


class TestCheckLinks:
    def test_rejects_untracked_target_even_if_present_on_disk(self, git_repo: Path) -> None:
        """The regression this module exists to prevent: a link target
        that exists on the filesystem but was never `git add`-ed must
        fail, because a fresh clone will not have it either."""
        _write(git_repo, "README.md", "[see](notes/local-only.md)\n")
        _git(git_repo, "add", "README.md")
        _write(git_repo, "notes/local-only.md", "# never committed\n")

        tracked = check_docs.tracked_files(git_repo)
        problems = check_docs.check_links(git_repo, tracked)

        assert len(problems) == 1
        assert "notes/local-only.md" in problems[0]

    def test_accepts_a_tracked_file_target(self, git_repo: Path) -> None:
        _write(git_repo, "README.md", "[see](docs/thing.md)\n")
        _write(git_repo, "docs/thing.md", "# thing\n")
        _git(git_repo, "add", "-A")

        tracked = check_docs.tracked_files(git_repo)
        problems = check_docs.check_links(git_repo, tracked)

        assert problems == []

    def test_accepts_a_tracked_directory_target(self, git_repo: Path) -> None:
        """Git has no directory objects: a link to `history/` is only
        valid because a file lives under it, never as its own entry in
        `git ls-files`."""
        _write(git_repo, "README.md", "[see](history/)\n")
        _write(git_repo, "history/note.md", "# a file making the dir real\n")
        _git(git_repo, "add", "-A")

        tracked = check_docs.tracked_files(git_repo)
        problems = check_docs.check_links(git_repo, tracked)

        assert problems == []

    def test_ignores_http_and_mailto_links(self, git_repo: Path) -> None:
        _write(
            git_repo,
            "README.md",
            "[ext](https://example.com/x)\n[mail](mailto:a@b.com)\n",
        )
        _git(git_repo, "add", "README.md")

        tracked = check_docs.tracked_files(git_repo)
        problems = check_docs.check_links(git_repo, tracked)

        assert problems == []

    def test_resolves_relative_to_the_linking_file_not_the_repo_root(self, git_repo: Path) -> None:
        _write(git_repo, "docs/guide.md", "[see](../examples/thing.md)\n")
        _write(git_repo, "examples/thing.md", "# thing\n")
        _git(git_repo, "add", "-A")

        tracked = check_docs.tracked_files(git_repo)
        problems = check_docs.check_links(git_repo, tracked)

        assert problems == []


class TestRequiredIndexes:
    def test_missing_tracked_index_is_flagged(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(check_docs, "REQUIRED_INDEXES", ["docs/README.md"])
        tracked = check_docs.tracked_files(git_repo)

        problems = check_docs.check_required_indexes(tracked)

        assert len(problems) == 1
        assert "docs/README.md" in problems[0]

    def test_an_untracked_on_disk_index_still_counts_as_missing(
        self, git_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mirrors the real incident: `evaluations/README.md` existed on
        disk but was silently dropped by a blanket `.gitignore` rule, so
        it had to count as missing until explicitly force-added."""
        monkeypatch.setattr(check_docs, "REQUIRED_INDEXES", ["evaluations/README.md"])
        _write(git_repo, "evaluations/README.md", "# present on disk, never added\n")

        tracked = check_docs.tracked_files(git_repo)
        problems = check_docs.check_required_indexes(tracked)

        assert len(problems) == 1

    def test_a_tracked_index_passes(self, git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(check_docs, "REQUIRED_INDEXES", ["docs/README.md"])
        _write(git_repo, "docs/README.md", "# index\n")
        _git(git_repo, "add", "-A")

        tracked = check_docs.tracked_files(git_repo)
        problems = check_docs.check_required_indexes(tracked)

        assert problems == []


class TestReadmeLength:
    def test_over_the_ceiling_is_flagged(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("\n".join(["line"] * (check_docs.README_MAX_LINES + 1)))

        problems = check_docs.check_readme_length(tmp_path)

        assert len(problems) == 1
        assert str(check_docs.README_MAX_LINES) in problems[0]

    def test_at_the_ceiling_passes(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("\n".join(["line"] * check_docs.README_MAX_LINES))

        assert check_docs.check_readme_length(tmp_path) == []


class TestNoUnscopedDeployedSha:
    def test_flags_a_present_tense_claim_outside_history(self, git_repo: Path) -> None:
        _write(
            git_repo,
            "docs/NOTES.md",
            "The deployed service runs `abc1234def` as of right now.\n",
        )
        _git(git_repo, "add", "docs/NOTES.md")

        tracked = check_docs.tracked_files(git_repo)
        problems = check_docs.check_no_unscoped_deployed_sha(git_repo, tracked)

        assert len(problems) == 1
        assert "docs/NOTES.md" in problems[0]

    def test_exempts_docs_history(self, git_repo: Path) -> None:
        _write(
            git_repo,
            "docs/history/releases/old.md",
            "The deployed service runs `abc1234def`.\n",
        )
        _git(git_repo, "add", "-A")

        tracked = check_docs.tracked_files(git_repo)
        problems = check_docs.check_no_unscoped_deployed_sha(git_repo, tracked)

        assert problems == []

    def test_exempts_examples_live_validation(self, git_repo: Path) -> None:
        _write(
            git_repo,
            "examples/live-validation/run-1/notes.md",
            "The deployed service runs `abc1234def`.\n",
        )
        _git(git_repo, "add", "-A")

        tracked = check_docs.tracked_files(git_repo)
        problems = check_docs.check_no_unscoped_deployed_sha(git_repo, tracked)

        assert problems == []

    def test_a_plain_commit_sha_citation_is_not_flagged(self, git_repo: Path) -> None:
        """This check is narrowly about the "currently deployed" framing,
        not about mentioning a commit SHA at all -- those are normal and
        frequent in this project's evidence-based docs."""
        _write(
            git_repo,
            "docs/NOTES.md",
            "Fixed in commit `abc1234def`, released as part of v1.9.0.\n",
        )
        _git(git_repo, "add", "docs/NOTES.md")

        tracked = check_docs.tracked_files(git_repo)
        problems = check_docs.check_no_unscoped_deployed_sha(git_repo, tracked)

        assert problems == []
