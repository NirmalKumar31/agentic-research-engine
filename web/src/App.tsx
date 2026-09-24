import { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchConfig,
  fetchExamples,
  replayExample,
  runResearch,
  type DemoConfig,
  type ExampleSummary,
} from "./api";
import { HeroDiagram } from "./HeroDiagram";
import { advance, emptyPipeline, humanise, STAGES } from "./pipelineState";
import { Pipeline } from "./Pipeline";
import { describe } from "./progress";
import { ReportView } from "./ReportView";
import type { ProgressEvent, RunResult } from "./types";

/** Capability badges, keyed by recording id rather than guessed from data. */
const BADGES: Record<string, string> = {
  "rag-vector-vs-search": "Multi-source research",
  "nist-ai-risk-framework": "PDF page provenance",
  "fraud-detection-imbalanced": "6 research dimensions",
};

export default function App() {
  const [config, setConfig] = useState<DemoConfig | null>(null);
  const [examples, setExamples] = useState<ExampleSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [feed, setFeed] = useState<string[]>([]);
  const [rawEvents, setRawEvents] = useState<string[]>([]);
  const [showRaw, setShowRaw] = useState(false);
  const [pipeline, setPipeline] = useState(emptyPipeline);
  const [result, setResult] = useState<RunResult | null>(null);
  const [error, setError] = useState<{ message: string; capacity: boolean } | null>(null);
  const [running, setRunning] = useState(false);
  const [isRecorded, setIsRecorded] = useState(false);

  const abort = useRef<AbortController | null>(null);
  const feedEnd = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    fetchConfig().then(setConfig);
    fetchExamples().then(setExamples);
  }, []);

  useEffect(() => {
    feedEnd.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [feed]);

  useEffect(() => () => abort.current?.abort(), []);

  const live = config?.live_research_enabled ?? false;
  const tooLong = config ? query.trim().length > config.max_query_chars : false;
  const remaining = config ? config.max_query_chars - query.trim().length : 0;

  const begin = (recorded: boolean) => {
    const controller = new AbortController();
    abort.current = controller;
    setRunning(true);
    setIsRecorded(recorded);
    setFeed([]);
    setRawEvents([]);
    setPipeline(emptyPipeline());
    setResult(null);
    setError(null);
    return controller;
  };

  const handlers = {
    onProgress: (event: ProgressEvent) => {
      setPipeline((p) => advance(p, event));
      const line = humanise(event);
      if (line) setFeed((prev) => [...prev, line]);
      const raw = describe(event);
      if (raw) setRawEvents((prev) => [...prev, `${event.event} — ${raw}`]);
    },
    onResult: setResult,
    onError: (message: string, capacity?: boolean) =>
      setError({ message, capacity: Boolean(capacity) }),
    onDone: () => setRunning(false),
  };

  const play = async (example: ExampleSummary) => {
    if (running) return;
    setActiveId(example.id);
    const controller = begin(true);
    try {
      await replayExample(example.id, handlers, controller.signal);
    } catch (exc) {
      if ((exc as Error).name !== "AbortError")
        setError({ message: "Could not load that recording.", capacity: false });
      setRunning(false);
    }
  };

  const startLive = async () => {
    const trimmed = query.trim();
    if (!trimmed || running) return;
    setActiveId(null);
    const controller = begin(false);
    try {
      await runResearch(trimmed, handlers, controller.signal);
    } catch (exc) {
      if ((exc as Error).name !== "AbortError")
        setError({ message: "Lost connection to the server.", capacity: false });
      setRunning(false);
    }
  };

  const stop = () => {
    abort.current?.abort();
    setRunning(false);
  };

  const showPipeline = running || pipeline.queries.length > 0 || feed.length > 0;
  const statusBadge = useMemo(
    () => (isRecorded ? { label: "Recorded", cls: "badge--recorded" } : { label: "Live", cls: "badge--live" }),
    [isRecorded],
  );

  return (
    <div className="app">
      <header className="masthead">
        <div className="mark" aria-hidden="true">
          <span />
        </div>
        <span className="masthead__name">Agentic Research Engine</span>
        <nav className="masthead__nav">
          <a href="#how">How it works</a>
          <a
            href="https://github.com/NirmalKumar31/agentic-research-engine"
            target="_blank"
            rel="noreferrer noopener"
          >
            GitHub ↗
          </a>
        </nav>
      </header>

      <section className="hero">
        <div className="hero__copy">
          <h1>Research that shows its work.</h1>
          <p className="hero__sub">
            Multi-stage agentic research with evidence-level provenance. Published claims
            link to the source passages that passed evidence verification.
          </p>
        </div>
        <HeroDiagram />
      </section>

      {live ? (
        <section className="ask">
          <label className="sr-only" htmlFor="q">
            Research question
          </label>
          <textarea
            id="q"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Compare the practical trade-offs between vector databases and traditional search for RAG…"
            rows={3}
            disabled={running}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) startLive();
            }}
          />
          <div className="ask__row">
            <span className="muted small">
              Live demo · 1 research round · up to {config?.max_sources ?? 6} sources
              {remaining < 60 && (
                <span className={tooLong ? "over" : ""}> · {remaining} characters left</span>
              )}
            </span>
            <span className="ask__actions">
              <kbd className="muted small">⌘/Ctrl + Enter</kbd>
              {running && !isRecorded ? (
                <button type="button" className="btn" onClick={stop}>
                  Stop
                </button>
              ) : (
                <button
                  type="button"
                  className="btn btn--primary"
                  onClick={startLive}
                  disabled={!query.trim() || tooLong || running}
                >
                  Research
                </button>
              )}
            </span>
          </div>
        </section>
      ) : (
        <section className="notice">
          <p>
            <strong>Live research is off on this instance.</strong> The daily run cap is held
            in process memory, and a host that sleeps when idle resets it on every cold start,
            so it cannot bound an API quota. Rather than add a database whose only job is
            letting strangers spend the budget, this site replays real recorded runs.{" "}
            <a
              href="https://github.com/NirmalKumar31/agentic-research-engine#usage"
              target="_blank"
              rel="noreferrer noopener"
            >
              Run it yourself
            </a>{" "}
            for live research — the models can run locally on Ollama with no paid LLM usage,
            though live web research still needs a search provider configured.
          </p>
        </section>
      )}

      <section className="examples">
        <div className="examples__head">
          <h2>{live ? "Or explore a recorded run" : "Recorded demonstrations"}</h2>
          <span className="muted small">Real executions, replayed from their own events</span>
        </div>
        <div className="cards">
          {examples.map((example) => (
            <button
              key={example.id}
              type="button"
              className={`card ${activeId === example.id ? "card--active" : ""}`}
              onClick={() => play(example)}
              disabled={running}
            >
              {BADGES[example.id] && <span className="card__badge">{BADGES[example.id]}</span>}
              <span className="card__title">{example.label}</span>
              <span className="card__desc">{example.description}</span>
              <span className="card__facts muted small">
                {example.sources} sources · {example.citable_evidence} citable ·{" "}
                {Math.round(example.duration_s)}s
              </span>
            </button>
          ))}
        </div>
      </section>

      {error && (
        <div className={`alert ${error.capacity ? "alert--capacity" : ""}`} role="alert">
          {error.message}
        </div>
      )}

      {showPipeline && (
        <section className="run">
          <div className="run__head">
            <span className={`badge ${statusBadge.cls}`}>
              {running && <span className="badge__pulse" aria-hidden="true" />}
              {statusBadge.label}
            </span>
            <span className="muted small">
              {running ? "Running" : pipeline.finished ? "Finished" : "Stopped"}
            </span>
          </div>

          <Pipeline state={pipeline} live={running} />

          {feed.length > 0 && (
            <div className="feed">
              <ol>
                {feed.map((line, i) => (
                  <li key={i}>{line}</li>
                ))}
              </ol>
              <div ref={feedEnd} />
              <button
                type="button"
                className="linkish muted small"
                onClick={() => setShowRaw(!showRaw)}
                aria-expanded={showRaw}
              >
                {showRaw ? "Hide" : "Show"} technical events ({rawEvents.length})
              </button>
              {showRaw && (
                <ol className="feed__raw">
                  {rawEvents.map((line, i) => (
                    <li key={i}>
                      <code>{line}</code>
                    </li>
                  ))}
                </ol>
              )}
            </div>
          )}
        </section>
      )}

      {result && <ReportView result={result} />}

      <section id="how" className="how">
        <h2>How it works</h2>
        <ol className="how__list">
          {STAGES.map((stage) => (
            <li key={stage.id}>
              <span className="how__label">{stage.label}</span>
              <span className="how__blurb muted">{stage.blurb}</span>
            </li>
          ))}
        </ol>
      </section>

      <section className="why">
        <h2>Why this is different</h2>
        <div className="why__grid">
          <div>
            <h3>Evidence-level provenance</h3>
            <p className="muted">
              Claims point to the exact supporting passage, not merely a URL. The model picks
              evidence ids; the engine derives citations from them.
            </p>
          </div>
          <div>
            <h3>Bounded agentic research</h3>
            <p className="muted">
              Parallel search and iterative coverage assessment, under hard request, token and
              spend ceilings checked before each call is dispatched.
            </p>
          </div>
          <div>
            <h3>Honest verification</h3>
            <p className="muted">
              A quote that could not be matched to its source cannot ground a citation. Dropped
              references are reported rather than quietly removed.
            </p>
          </div>
        </div>
      </section>

      <footer className="footer">
        <nav>
          <a
            href="https://github.com/NirmalKumar31/agentic-research-engine"
            target="_blank"
            rel="noreferrer noopener"
          >
            GitHub
          </a>
          <a
            href="https://github.com/NirmalKumar31/agentic-research-engine/blob/main/docs/ARCHITECTURE.md"
            target="_blank"
            rel="noreferrer noopener"
          >
            Architecture
          </a>
          <a
            href="https://github.com/NirmalKumar31/agentic-research-engine/blob/main/docs/LIMITATIONS.md"
            target="_blank"
            rel="noreferrer noopener"
          >
            Limitations
          </a>
        </nav>
        <p className="muted small">
          Verifies faithfulness to retrieved evidence, not truth about the world.
        </p>
      </footer>
    </div>
  );
}
