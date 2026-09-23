"""Terminal rendering of run progress.

Consumes the same event stream the Streamlit app and the evaluation harness
consume. Presentation lives here; nothing in the graph knows a terminal
exists.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

from agentic_research.metrics import RunMetrics


class ProgressPrinter:
    """Renders progress events as readable lines."""

    def __init__(self, console: Console, verbose: bool = False) -> None:
        self.console = console
        self.verbose = verbose
        self._searches = 0
        self._sources = 0
        self._evidence = 0

    def handle(self, event: dict[str, Any]) -> None:
        name = event.get("event", "")
        handler = getattr(self, f"_on_{name}", None)
        if handler is not None:
            handler(event)
        elif self.verbose:
            self.console.print(f"[dim]{name}: {event}[/dim]")

    # -- lifecycle ---------------------------------------------------------

    def _on_started(self, event: dict[str, Any]) -> None:
        self.console.print(f"[bold]Researching:[/bold] {event['query']}")
        self.console.print(f"[dim]run {event['run_id']}[/dim]")
        models = event.get("models", {})
        if models:
            summary = ", ".join(f"{role}={spec}" for role, spec in sorted(models.items()))
            self.console.print(f"[dim]{summary}[/dim]\n")

    def _on_warning(self, event: dict[str, Any]) -> None:
        self.console.print(f"[yellow]warning[/yellow] {event['message']}")

    # -- planning ----------------------------------------------------------

    def _on_analyzing_query(self, _: dict[str, Any]) -> None:
        self.console.print("Analysing question...")

    def _on_query_analyzed(self, event: dict[str, Any]) -> None:
        self.console.print(f"  intent: [cyan]{event.get('intent', '')}[/cyan]")

    def _on_plan_generated(self, event: dict[str, Any]) -> None:
        self.console.print(f"Research plan: {event['count']} sub-questions")
        for question in event.get("questions", []):
            self.console.print(f"  [dim]-[/dim] {question}")

    def _on_queries_generated(self, event: dict[str, Any]) -> None:
        self.console.print(f"\nRound {event['round']}: searching {event['count']} queries")
        if self.verbose:
            for query in event.get("queries", []):
                self.console.print(f"  [dim]?[/dim] {query}")

    # -- research ----------------------------------------------------------

    def _on_search_completed(self, event: dict[str, Any]) -> None:
        self._searches += 1
        if self.verbose:
            self.console.print(
                f"  [green]ok[/green] {event['query'][:60]} -> {event['results']} results"
            )

    def _on_search_failed(self, event: dict[str, Any]) -> None:
        self.console.print(f"  [red]search failed[/red] {event.get('query', '')[:60]}")

    def _on_sources_deduplicated(self, event: dict[str, Any]) -> None:
        self.console.print(
            f"  {event['results']} results -> {event['unique']} unique "
            f"([green]{event['avoided']}[/green] duplicate fetches avoided), "
            f"retrieving {event['selected']}"
        )

    def _on_source_retrieved(self, event: dict[str, Any]) -> None:
        self._sources += 1
        if self.verbose:
            self.console.print(f"  [dim]+[/dim] [{event['source_id']}] {event['title']}")

    def _on_source_failed(self, event: dict[str, Any]) -> None:
        if self.verbose:
            self.console.print(
                f"  [yellow]-[/yellow] [{event['source_id']}] unusable ({event['status']})"
            )

    def _on_sources_registered(self, event: dict[str, Any]) -> None:
        self.console.print(
            f"  {event['usable']} usable sources, extracting evidence from {event['extracting']}"
        )

    def _on_evidence_extracted(self, event: dict[str, Any]) -> None:
        self._evidence += event.get("items", 0)

    def _on_coverage_evaluated(self, event: dict[str, Any]) -> None:
        verdict = (
            "[green]sufficient[/green]" if event["sufficient"] else "[yellow]gaps remain[/yellow]"
        )
        self.console.print(
            f"  coverage {event['ratio']:.0%} ({event['covered']} covered, "
            f"{event['weak']} weak, {event['missing']} missing) - {verdict}"
        )

    def _on_followups_generated(self, event: dict[str, Any]) -> None:
        self.console.print(f"  following up on {event['count']} gap(s)")
        for question in event.get("questions", []):
            self.console.print(f"    [dim]-[/dim] {question}")

    # -- reporting ---------------------------------------------------------

    def _on_synthesizing(self, event: dict[str, Any]) -> None:
        self.console.print(f"\nSynthesising from {event['evidence']} evidence items...")

    def _on_verifying_citations(self, _: dict[str, Any]) -> None:
        self.console.print("Verifying citations...")

    def _on_citations_verified(self, event: dict[str, Any]) -> None:
        repaired = " [yellow](repaired)[/yellow]" if event.get("repaired") else ""
        self.console.print(
            f"  {event['valid']}/{event['total']} citations resolve to retrieved sources{repaired}"
        )

    def _on_completed(self, event: dict[str, Any]) -> None:
        self.console.print(f"\n[bold green]Complete[/bold green] ({event['stop_reason']})")


def metrics_table(metrics: RunMetrics) -> Table:
    table = Table(title="Run metrics", show_header=False, title_style="bold")
    table.add_column("metric", style="dim")
    table.add_column("value")

    rows: list[tuple[str, str]] = [
        ("Mode", metrics.mode),
        ("Research rounds", str(metrics.research_rounds)),
        ("Stopped because", metrics.stop_reason or "coverage sufficient"),
        ("Search queries", str(metrics.search_queries)),
        ("Unique sources", f"{metrics.unique_sources} ({metrics.usable_sources} usable)"),
        ("Distinct domains", str(metrics.distinct_domains)),
        ("Duplicate fetches avoided", str(metrics.fetches_avoided)),
        ("Evidence items", str(metrics.evidence_items)),
        ("Quotes verbatim (exact)", f"{metrics.quote_fidelity_rate:.0%}"),
        ("Quotes fuzzy (not citable)", f"{metrics.fuzzy_quote_rate:.0%}"),
        ("LLM calls", str(metrics.llm_calls)),
        ("Tokens in/out", f"{metrics.input_tokens:,} / {metrics.output_tokens:,}"),
        ("Estimated cost", metrics.cost_display),
        ("Duration", f"{metrics.duration_s:.1f}s"),
    ]
    if metrics.evidence_integrity_rate is not None:
        rows.insert(
            9,
            (
                "Evidence integrity",
                f"{metrics.evidence_integrity_rate:.0%} of "
                f"{metrics.evidence_refs_total} references resolved",
            ),
        )
    if metrics.claim_support_rate is not None:
        breakdown = metrics.support_breakdown
        scope = "exhaustive" if metrics.entailment_exhaustive else "sampled"
        rows.insert(
            10,
            (
                f"Claim support ({scope})",
                f"{breakdown.get('supported', 0)} supported / "
                f"{breakdown.get('partially_supported', 0)} partial / "
                f"{breakdown.get('unsupported', 0)} unsupported / "
                f"{breakdown.get('not_checked', 0)} unchecked",
            ),
        )
    if metrics.content_origins:
        origins = ", ".join(f"{k}={v}" for k, v in sorted(metrics.content_origins.items()))
        rows.append(("Content origin", origins))
    if metrics.errors:
        rows.append(("Recoverable errors", str(metrics.errors)))

    for label, value in rows:
        table.add_row(label, value)
    return table
