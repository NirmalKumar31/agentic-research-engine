"""Command-line interface."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from agentic_research.config import LLMMode, Settings, get_settings
from agentic_research.llm.base import ModelUnavailableError
from agentic_research.observability import configure_logging
from agentic_research.runner import RunResult, new_run_id, stream_research
from agentic_research.search.service import SearchProviderNotConfigured

app = typer.Typer(
    name="agentic-research",
    help="A LangGraph deep-research engine with evidence provenance and citation verification.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


def _load_settings(**overrides: object) -> Settings:
    """Layer CLI flags on top of the environment.

    The CLI overrides ``.env``, which overrides defaults. Flags are applied by
    constructing a fresh Settings so validation runs against the final values
    rather than a half-updated object.
    """
    get_settings.cache_clear()
    base = get_settings()
    applied = {k: v for k, v in overrides.items() if v is not None}
    if not applied:
        return base
    return Settings(**{**base.model_dump(), **applied})


def _fail(message: str, hint: str = "") -> None:
    err_console.print(f"[bold red]Error:[/bold red] {message}")
    if hint:
        err_console.print(f"[dim]{hint}[/dim]")
    raise typer.Exit(code=1)


@app.command()
def research(
    question: Annotated[str, typer.Argument(help="The research question.")],
    mode: Annotated[
        LLMMode | None,
        typer.Option("--mode", "-m", help="cloud, local or hybrid. Overrides LLM_MODE."),
    ] = None,
    max_rounds: Annotated[
        int | None, typer.Option("--max-rounds", help="Cap on research rounds.")
    ] = None,
    max_sources: Annotated[
        int | None, typer.Option("--max-sources", help="Cap on sources retrieved.")
    ] = None,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write the report to this file.")
    ] = None,
    show_report: Annotated[
        bool, typer.Option("--show/--no-show", help="Print the report to the terminal.")
    ] = True,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show every query and source.")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="Suppress progress; print only the report.")
    ] = False,
) -> None:
    """Research a question and produce an evidence-backed report."""
    from agentic_research.cli.progress import ProgressPrinter, metrics_table

    try:
        settings = _load_settings(
            llm_mode=mode, max_research_rounds=max_rounds, max_sources=max_sources
        )
    except Exception as exc:
        _fail(str(exc), "Check your .env against .env.example.")
        return

    configure_logging(
        "WARNING" if quiet else ("DEBUG" if verbose else settings.log_level),
        settings.log_format,
    )

    printer = ProgressPrinter(console, verbose=verbose)
    run_id = new_run_id()

    async def drive() -> RunResult | None:
        result: RunResult | None = None
        async for event in stream_research(question, settings, run_id=run_id):
            if event.get("event") == "result":
                result = event["result"]
            elif not quiet:
                printer.handle(event)
        return result

    try:
        result = asyncio.run(drive())
    except ModelUnavailableError as exc:
        _fail(str(exc))
        return
    except SearchProviderNotConfigured as exc:
        _fail(str(exc))
        return
    except KeyboardInterrupt:
        err_console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130) from None

    if result is None:
        _fail("The run produced no result.")
        return

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.markdown, encoding="utf-8")
        console.print(f"\nReport written to [bold]{output}[/bold]")

    if show_report and not quiet:
        console.print()
        console.print(Markdown(result.markdown))
    elif quiet:
        sys.stdout.write(result.markdown)

    if not quiet:
        console.print()
        console.print(metrics_table(result.metrics))
        if result.output_dir is not None:
            console.print(f"[dim]Artifacts: {result.output_dir}[/dim]")

    # A report whose citations could not all be resolved is a degraded result
    # and should be detectable by a script, not only by reading the output.
    verification = result.state.get("verification") or {}
    if any(issue.get("severity") == "error" for issue in verification.get("issues", [])):
        raise typer.Exit(code=2)


@app.command()
def check() -> None:
    """Verify configuration and provider reachability without running research."""
    from agentic_research.llm.router import ModelRouter

    try:
        settings = _load_settings()
    except Exception as exc:
        _fail(str(exc), "Check your .env against .env.example.")
        return

    configure_logging("WARNING", settings.log_format)
    table = Table(title="Configuration", show_header=True)
    table.add_column("Check")
    table.add_column("Result")

    table.add_row("Mode", settings.llm_mode.value)
    for role, spec in sorted(ModelRouter(settings).describe().items()):
        table.add_row(f"  {role}", spec)

    table.add_row("Search provider", settings.search_provider)
    table.add_row(
        "Tavily key", "[green]set[/green]" if settings.tavily_api_key else "[red]missing[/red]"
    )
    table.add_row(
        "OpenAI key", "[green]set[/green]" if settings.openai_api_key else "[dim]not set[/dim]"
    )

    router = ModelRouter(settings)
    try:
        warnings = asyncio.run(router.preflight())
        table.add_row("Model preflight", "[green]ok[/green]")
        for warning in warnings:
            table.add_row("  warning", f"[yellow]{warning}[/yellow]")
    except ModelUnavailableError as exc:
        table.add_row("Model preflight", f"[red]{exc}[/red]")

    table.add_row(
        "Budgets",
        f"{settings.max_research_rounds} rounds, "
        f"{settings.max_search_queries} queries, "
        f"{settings.max_sources} sources, "
        f"{settings.max_llm_calls} LLM calls",
    )
    table.add_row("Checkpointing", settings.checkpoint_backend)
    console.print(table)


@app.command("show")
def show_run(
    run_id: Annotated[str, typer.Argument(help="Run id, or 'latest'.")] = "latest",
) -> None:
    """Re-display a stored run without re-running any research."""
    settings = _load_settings()
    directory = settings.output_dir
    if not directory.is_dir():
        _fail(f"No output directory at {directory}")
        return

    if run_id == "latest":
        runs = sorted(p for p in directory.iterdir() if p.is_dir())
        if not runs:
            _fail(f"No stored runs in {directory}")
            return
        target = runs[-1]
    else:
        target = directory / run_id
        if not target.is_dir():
            _fail(f"No such run: {run_id}")
            return

    report = target / "report.md"
    if report.is_file():
        console.print(Markdown(report.read_text(encoding="utf-8")))
    metrics_file = target / "metrics.json"
    if metrics_file.is_file():
        from agentic_research.cli.progress import metrics_table
        from agentic_research.metrics import RunMetrics

        console.print()
        console.print(
            metrics_table(RunMetrics.model_validate(json.loads(metrics_file.read_text())))
        )


@app.command()
def evaluate(
    limit: Annotated[
        int | None, typer.Option("--limit", "-n", help="Run only the first N questions.")
    ] = None,
    mode: Annotated[LLMMode | None, typer.Option("--mode", "-m", help="Override LLM_MODE.")] = None,
    output_dir: Annotated[
        Path, typer.Option("--output-dir", help="Where to write the benchmark JSON.")
    ] = Path("evaluations"),
) -> None:
    """Run the benchmark question set and report measured quality metrics.

    This spends real API credits. It is not part of the test suite.
    """
    from agentic_research.evaluation import BENCHMARK, run_benchmark, write_report

    try:
        settings = _load_settings(llm_mode=mode)
    except Exception as exc:
        _fail(str(exc))
        return

    configure_logging("WARNING", settings.log_format)
    total = min(limit, len(BENCHMARK)) if limit else len(BENCHMARK)
    console.print(f"Running {total} benchmark question(s) in {settings.llm_mode.value} mode.\n")

    def announce(question: object) -> None:
        console.print(f"[bold]{question.id}[/bold] {question.question}")  # type: ignore[attr-defined]

    report = asyncio.run(run_benchmark(settings, limit=limit, on_question=announce))

    table = Table(title="Benchmark results", show_header=True)
    table.add_column("Metric")
    table.add_column("Mean", justify="right")
    for name, value in report.aggregate().items():
        table.add_row(name, "n/a" if value is None else f"{value:.1%}")
    console.print()
    console.print(table)

    totals = report.totals()
    console.print(
        f"\n{totals['succeeded']}/{totals['questions']} runs succeeded | "
        f"{totals['total_sources']} sources | {totals['total_llm_calls']} LLM calls | "
        f"${totals['total_cost_usd']:.4f} | mean {totals['mean_duration_s']}s per question"
    )
    path = write_report(report, output_dir)
    console.print(f"[dim]Written to {path}[/dim]")


@app.command("graph")
def show_graph(
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Write Mermaid source to a file.")
    ] = None,
) -> None:
    """Print the research graph as a Mermaid diagram."""
    from agentic_research.graph.workflow import render_mermaid

    mermaid = render_mermaid()
    if output is not None:
        output.write_text(mermaid, encoding="utf-8")
        console.print(f"Written to {output}")
    else:
        # Straight to stdout: Rich would treat Mermaid's [node] labels as
        # markup tags and silently strip them, and this output is meant to be
        # copied into a document verbatim.
        sys.stdout.write(mermaid + "\n")


if __name__ == "__main__":
    app()
