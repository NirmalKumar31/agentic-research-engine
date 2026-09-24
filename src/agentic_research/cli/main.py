"""Command-line interface."""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from agentic_research.config import LLMMode, Settings, get_settings
from agentic_research.environment import capture as capture_environment
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


@app.command("freeze")
def freeze_corpus(
    question: Annotated[str, typer.Argument(help="Question to research and freeze.")],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Where to write the corpus.")
    ] = Path("evaluations/corpus.json"),
    mode: Annotated[LLMMode | None, typer.Option("--mode", "-m")] = None,
) -> None:
    """Run research once and freeze its evidence for controlled comparison.

    Replaying synthesis against a frozen corpus is what makes a model
    comparison valid: both arms then see identical evidence, so the
    difference is the model rather than the search results having moved.
    """
    from agentic_research.evaluation.ab import EvidenceCorpus
    from agentic_research.runner import run_research

    settings = _load_settings(llm_mode=mode)
    configure_logging(settings.log_level, settings.log_format)
    console.print(f"Researching to build a corpus: [bold]{question}[/bold]")

    result = asyncio.run(run_research(question, settings))
    corpus = EvidenceCorpus.from_result(question, result)
    corpus.save(output)
    console.print(f"Frozen: {corpus.summary()}")
    console.print(f"[dim]Written to {output}[/dim]")


@app.command("compare")
def compare_models(
    corpus_path: Annotated[Path, typer.Argument(help="Corpus produced by `freeze`.")] = Path(
        "evaluations/corpus.json"
    ),
    arm: Annotated[
        list[str] | None,
        typer.Option(
            "--arm",
            "-a",
            help="label=provider:model, repeatable. e.g. -a local=ollama:qwen3:4b",
        ),
    ] = None,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("evaluations/comparison.json"),
) -> None:
    """Synthesise the same frozen corpus with different models and compare.

    Spends real credits for any cloud arm. Verification is exhaustive.
    """
    from agentic_research.evaluation.ab import EvidenceCorpus, all_roles, compare

    if not corpus_path.is_file():
        _fail(f"No corpus at {corpus_path}", "Create one with `agentic-research freeze`.")
        return
    if not arm:
        _fail("At least one --arm is required, e.g. -a local=ollama:qwen3:4b")
        return

    arms: dict[str, dict] = {}
    for entry in arm:
        if "=" not in entry:
            _fail(f"Malformed --arm {entry!r}; expected label=provider:model")
            return
        label, spec = entry.split("=", 1)
        try:
            arms[label.strip()] = all_roles(spec.strip())
        except ValueError as exc:
            _fail(str(exc))
            return

    settings = _load_settings()
    configure_logging("WARNING", settings.log_format)
    corpus = EvidenceCorpus.load(corpus_path)
    problems = corpus.validate_for_replay()
    if problems:
        # Refuse rather than print a table whose zeros look like findings.
        _fail("This corpus cannot support verification:\n  - " + "\n  - ".join(problems))
        return
    console.print(f"Corpus: {corpus.summary()}")
    console.print(f"Question: {corpus.question}\n")

    comparison = asyncio.run(compare(corpus, settings, arms))

    table = Table(title="Controlled comparison (identical evidence)", show_header=True)
    table.add_column("Metric")
    for result in comparison.arms:
        table.add_column(result.label, justify="right")

    names: list[str] = []
    for result in comparison.arms:
        for metric in result.metrics:
            if metric.name not in names:
                names.append(metric.name)
    for name in names:
        row = [name]
        for result in comparison.arms:
            value = result.value(name)
            row.append("n/a" if value is None else f"{value:.1%}")
        table.add_row(*row)

    for label, getter in (
        ("duration_s", lambda a: f"{a.duration_s:.1f}"),
        ("llm_calls", lambda a: str(a.llm_calls)),
        ("input_tokens", lambda a: f"{a.input_tokens:,}"),
        ("output_tokens", lambda a: f"{a.output_tokens:,}"),
        ("cost_usd", lambda a: f"${a.known_cost_usd:.4f}"),
    ):
        table.add_row(label, *[getter(a) for a in comparison.arms])

    console.print(table)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(comparison.to_dict(), indent=2, default=str), encoding="utf-8")
    console.print(f"[dim]Written to {output}[/dim]")


# Metrics reported as shares rather than counts, so the table formats them as
# percentages instead of printing 0.7391.
_RATE_METRICS = frozenset(
    {
        "quote_fidelity",
        "quote_drift",
        "cross_attribution_rate",
        "citable_cross_attribution_rate",
        "evidence_coverage",
        "sub_questions_with_citable_evidence",
        "source_utilisation",
    }
)

_ATTRIBUTION_ROWS = (
    ("evidence_items", "evidence items"),
    ("citable_items", "citable items"),
    ("quote_fidelity", "quote fidelity"),
    ("cross_attribution_rate", "cross-attributed"),
    ("citable_cross_attribution_rate", "cross-attributed (citable)"),
    ("evidence_coverage", "evidence coverage"),
    ("sub_questions_with_citable_evidence", "sub-questions answered"),
    ("source_utilisation", "sources earning their fetch"),
    ("mean_sub_questions_shown", "mean sub-questions shown"),
    ("extraction_calls", "extraction calls"),
    ("input_tokens", "input tokens"),
    ("output_tokens", "output tokens"),
    ("known_cost_usd", "cost"),
    ("duration_s", "duration (s)"),
)


def _format_measurement(name: str, value: float) -> str:
    if name in _RATE_METRICS:
        return f"{value:.1%}"
    if name == "known_cost_usd":
        return f"${value:.4f}"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _attribution_cell(name: str, aggregate: dict) -> str:
    """One table cell: the mean, with the observed range when repeats differ."""
    entry = aggregate.get(name)
    if not isinstance(entry, dict):
        return "n/a"
    text = _format_measurement(name, entry["mean"])
    if entry["n"] > 1 and entry["min"] != entry["max"]:
        low = _format_measurement(name, entry["min"])
        high = _format_measurement(name, entry["max"])
        text = f"{text} [{low}..{high}]"
    return text


@app.command("attribution")
def attribution_experiment(
    corpus_path: Annotated[Path, typer.Argument(help="Corpus produced by `freeze`.")] = Path(
        "evaluations/corpus.json"
    ),
    strategy: Annotated[
        list[str] | None,
        typer.Option(
            "--strategy",
            "-s",
            help="all_open, retrieved_only or adjacent. Repeatable; default is all three.",
        ),
    ] = None,
    repeats: Annotated[
        int, typer.Option("--repeats", "-r", help="Passes per strategy. One pass is one draw.")
    ] = 1,
    adjacent_k: Annotated[
        int, typer.Option("--adjacent-k", help="Extra nearest sub-questions for `adjacent`.")
    ] = 2,
    allow_cloud: Annotated[
        bool, typer.Option("--allow-cloud", help="Accept paid extraction calls.")
    ] = False,
    output: Annotated[Path, typer.Option("--output", "-o")] = Path("evaluations/attribution.json"),
) -> None:
    """Measure what narrowing extraction costs in evidence and coverage.

    Re-extracts from a frozen corpus under each strategy, varying only the
    sub-questions the extractor is shown. Free on a local model; refuses to
    run on a cloud one unless `--allow-cloud` is passed, because one pass is
    one call per source and three strategies multiply that quietly.
    """
    from agentic_research.evaluation.ab import EvidenceCorpus
    from agentic_research.evaluation.attribution import (
        CloudSpendRefused,
        Strategy,
        UnusableCorpusError,
        run_experiment,
        validate_corpus_for_extraction,
    )

    if not corpus_path.is_file():
        _fail(f"No corpus at {corpus_path}", "Create one with `agentic-research freeze`.")
        return

    try:
        chosen = [Strategy(name.strip()) for name in strategy] if strategy else list(Strategy)
    except ValueError as exc:
        _fail(str(exc), f"Valid strategies: {', '.join(s.value for s in Strategy)}")
        return

    settings = _load_settings()
    configure_logging("WARNING", settings.log_format)
    corpus = EvidenceCorpus.load(corpus_path)

    problems = validate_corpus_for_extraction(corpus)
    if problems:
        _fail(
            "This corpus cannot support the experiment:\n  - " + "\n  - ".join(problems),
            "A comparison run on it would produce zeros that read like findings.",
        )
        return

    console.print(f"Corpus: {corpus.summary()}")
    console.print(f"Question: {corpus.question}")
    console.print(
        f"Strategies: {', '.join(f'{s.letter}={s.value}' for s in chosen)} | {repeats} repeat(s)\n"
    )

    try:
        experiment = asyncio.run(
            run_experiment(
                corpus,
                settings,
                strategies=chosen,
                repeats=repeats,
                adjacent_k=adjacent_k,
                allow_cloud=allow_cloud,
            )
        )
    except CloudSpendRefused as exc:
        _fail(str(exc))
        return
    except UnusableCorpusError as exc:
        _fail(str(exc))
        return

    table = Table(
        title=f"Extraction strategy comparison ({experiment.eligible_sources} shared sources)",
        show_header=True,
    )
    table.add_column("Metric")
    for item in chosen:
        table.add_column(f"{item.letter}  {item.value}", justify="right")

    aggregates = {item.value: experiment.aggregate(item) for item in chosen}
    for name, label in _ATTRIBUTION_ROWS:
        table.add_row(label, *[_attribution_cell(name, aggregates[i.value]) for i in chosen])
    console.print(table)

    if experiment.excluded_sources:
        console.print(
            f"[dim]{len(experiment.excluded_sources)} source(s) excluded from every arm: "
            "no discovery path, so B and C would have had nothing to show them.[/dim]"
        )
    if repeats == 1:
        console.print(
            "[yellow]One pass per strategy is one draw from a sampling model. "
            "Use -r 3 or more before reading a difference as real.[/yellow]"
        )
    if experiment.breaches:
        # A breach means the harness measured something other than what it
        # claims, so the table above should not be read at all.
        console.print("\n[bold red]Invariant breached — do not trust these numbers:[/bold red]")
        for breach in experiment.breaches:
            console.print(f"  [red]- {breach}[/red]")

    console.print(
        "\n[dim]`retrieved_only` reaches 0% cross-attribution by construction. "
        "What the table measures is what that costs.[/dim]"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(experiment.to_dict(), indent=2, default=str), encoding="utf-8")
    console.print(f"[dim]Written to {output}[/dim]")

    if experiment.breaches:
        raise typer.Exit(code=2)


@app.command("record")
def record_example(
    question: Annotated[str, typer.Argument(help="Question to research and record.")],
    example_id: Annotated[
        str, typer.Option("--id", help="Lowercase slug; becomes the URL path segment.")
    ],
    label: Annotated[str, typer.Option("--label", help="Short title for the homepage card.")],
    description: Annotated[str, typer.Option("--description", help="One line of context.")] = "",
    order: Annotated[int, typer.Option("--order", help="Display order on the homepage.")] = 50,
    mode: Annotated[LLMMode | None, typer.Option("--mode", "-m")] = None,
    max_rounds: Annotated[int | None, typer.Option("--max-rounds")] = None,
    max_sources: Annotated[int | None, typer.Option("--max-sources")] = None,
) -> None:
    """Run research once and save it as a recorded demo for the website.

    The public site replays these instead of running live research, so the
    hosted demo cannot spend API credit. Captures the run's own progress
    events alongside the result: the replay shows what actually happened,
    never an invented sequence of stages.

    Source page text is excluded, so a recording is safe to commit.
    """
    from agentic_research.web.recordings import (
        RECORDING_SCHEMA_VERSION,
        RECORDINGS_DIR,
        assert_no_secrets,
        public_provenance,
        sanitise_trace,
        serialise_result,
        valid_id,
    )

    if not valid_id(example_id):
        _fail(
            f"Invalid --id {example_id!r}.",
            "Lowercase letters, digits and dashes; must start with a letter or digit.",
        )
        return

    settings = _load_settings(
        llm_mode=mode, max_research_rounds=max_rounds, max_sources=max_sources
    )
    configure_logging("WARNING", settings.log_format)
    console.print(f"Recording [bold]{example_id}[/bold]: {question}")

    # Captured before the run, not after. It has to describe the code that
    # is about to execute; taking it at the end would describe whatever the
    # tree looked like twenty minutes later, which is not the same thing.
    provenance = public_provenance(capture_environment(settings))
    if provenance.get("dirty"):
        _fail(
            "Refusing to record from a dirty working tree.",
            "A published recording must be reproducible from committed code. "
            "Commit or stash first.",
        )
        return

    trace: list[dict[str, Any]] = []
    result: RunResult | None = None

    async def drive() -> None:
        nonlocal result
        async for event in stream_research(
            question,
            settings,
            run_id=new_run_id(),
            # Canonical recordings are verified exhaustively: every eligible
            # claim is checked, and the publication gate can only remove a
            # claim it has actually assessed. Sampling would leave unchecked
            # claims in a published demo.
            exhaustive_verification=True,
        ):
            if event.get("event") == "result":
                result = event["result"]
            else:
                # The graph's own events, kept verbatim. The replay endpoint
                # emits exactly these, so nothing in the UI is invented.
                trace.append(event)
                console.print(f"  [dim]{event.get('event')}[/dim]")

    asyncio.run(drive())
    if result is None:
        _fail("The run produced no result; nothing recorded.")
        return

    serialised = serialise_result(result)
    payload: dict[str, Any] = {
        "recording_schema_version": RECORDING_SCHEMA_VERSION,
        "meta": {
            "id": example_id,
            "label": label,
            "question": question,
            "description": description,
            "mode": settings.llm_mode.value,
            "recorded_at": datetime.now(UTC).isoformat(),
            "order": order,
            # Identifiers only. A full environment capture holds resolved
            # settings, package versions and local model configuration --
            # right for a private artifact, wrong for a public file.
            "provenance": provenance,
        },
        "trace": sanitise_trace(trace),
        "result": serialised,
    }

    # Scanned before it is written, not after it is committed. The loader
    # checks this too, but by then the file exists and a `git add -A` has
    # had its chance -- and a credential caught in CI is a credential
    # already in history.
    try:
        assert_no_secrets(example_id, payload)
    except ValueError as exc:
        _fail(str(exc), "Nothing was written. Fix the source of that value first.")
        return

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    destination = RECORDINGS_DIR / f"{example_id}.json"
    destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    evidence = serialised.get("evidence", [])
    citable = sum(1 for e in evidence if e.get("citable"))
    with_pages = sum(1 for e in evidence if e.get("page"))
    console.print(
        f"\nRecorded {len(trace)} events, {len(evidence)} evidence items "
        f"({citable} citable, {with_pages} with a page number)."
    )
    console.print(f"[dim]Written to {destination}[/dim]")


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
