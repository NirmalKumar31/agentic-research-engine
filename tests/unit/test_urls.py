"""URL canonicalisation.

These cases are the ones that actually showed up in search results: tracking
parameters, AMP mirrors, mixed case hosts and default ports.
"""

from __future__ import annotations

import pytest

from agentic_research.retrieval.urls import canonicalize, domain_of, same_document


class TestCanonicalize:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("https://www.Example.com/post", "https://example.com/post"),
            ("https://example.com:443/post", "https://example.com/post"),
            ("http://example.com:80/post", "http://example.com/post"),
            ("https://example.com/post/", "https://example.com/post"),
            ("https://example.com/", "https://example.com/"),
            ("https://example.com/post#section-2", "https://example.com/post"),
            ("https://example.com/docs/index.html", "https://example.com/docs"),
            ("https://example.com/news/amp", "https://example.com/news"),
            ("https://example.com/p?utm_source=x&id=7", "https://example.com/p?id=7"),
            ("https://example.com/p?fbclid=abc", "https://example.com/p"),
            ("https://example.com/p?b=2&a=1", "https://example.com/p?a=1&b=2"),
        ],
    )
    def test_canonical_forms(self, raw: str, expected: str) -> None:
        assert canonicalize(raw) == expected

    def test_four_spellings_of_one_article_collapse(self) -> None:
        variants = [
            "https://www.Example.com/post?utm_source=x&id=7#intro",
            "https://example.com/post/?id=7",
            "http://example.com:80/post?id=7",
            "https://example.com/post?id=7&fbclid=abc",
        ]
        canonical = {canonicalize(v) for v in variants}
        # http and https are deliberately kept distinct; the rest collapse.
        assert canonical == {"https://example.com/post?id=7", "http://example.com/post?id=7"}

    def test_meaningful_params_survive_and_are_sorted(self) -> None:
        # Sorting is the point: ?q=..&page=.. and ?page=..&q=.. are one page.
        assert canonicalize("https://example.com/s?q=fraud&page=2") == (
            "https://example.com/s?page=2&q=fraud"
        )

    def test_distinct_subdomains_stay_distinct(self) -> None:
        assert canonicalize("https://docs.example.com/x") != canonicalize(
            "https://blog.example.com/x"
        )

    @pytest.mark.parametrize("bad", ["", "   ", "not a url", "mailto:a@b.com"])
    def test_unparseable_input_does_not_raise(self, bad: str) -> None:
        canonicalize(bad)  # must not raise


class TestDomain:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://www.example.com/a", "example.com"),
            ("https://docs.python.org/3/", "docs.python.org"),
            ("not a url", ""),
        ],
    )
    def test_domain_of(self, url: str, expected: str) -> None:
        assert domain_of(url) == expected

    def test_same_document(self) -> None:
        assert same_document("https://a.com/x?utm_source=q", "https://www.a.com/x/")
        assert not same_document("https://a.com/x", "https://a.com/y")
