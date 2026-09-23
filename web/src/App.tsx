import { useEffect, useRef, useState } from "react";
import {
  fetchConfig,
  fetchExamples,
  replayExample,
  runResearch,
  type DemoConfig,
  type ExampleSummary,
} from "./api";
import { describe, stageOf } from "./progress";
import { ReportView } from "./ReportView";
import type { ProgressEvent, RunResult } from "./types";

/**
 * Replay-first.
 *
 * The public instance serves recorded runs rather than executing new ones:
 * its daily cap lives in process memory and a host that spins down resets
 * it on every cold start, so it cannot bound an account-level quota.
 *
 * That is not a downgrade for a portfolio. The thing worth showing is the
 * provenance drill-down -- claim to evidence to verbatim quote to page to
 * source -- and a recording demonstrates it exactly as a live run would.
 * What matters is that it never pretends to be live.
 */
export default function App() {
  const [config, setConfig] = useState<DemoConfig | null>(null);
  const [examples, setExamples] = useState<ExampleSummary[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  const [query, setQuery] = useState("");
  const [lines, setLines] = useState<string[]>([]);
  const [stage, setStage] = useState("");
  const [result, setResult] = useState<RunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [isRecorded, setIsRecorded] = useState(false);

  const abort = useRef<AbortController | null>(null);
  const logEnd = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    fetchConfig().then(setConfig);
    fetchExamples().then(setExamples);
  }, []);

  useEffect(() => {
    logEnd.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [lines]);

  // Abort in-flight work if the component goes away, so a live run releases
  // its capacity slot rather than waiting for the server timeout.
  useEffect(() => () => abort.current?.abort(), []);

  const liveEnabled = config?.live_research_enabled ?? false;

  const beginStream = () => {
    const controller = new AbortController();
    abort.current = controller;
    setRunning(true);
    setLines([]);
    setResult(null);
    setError(null);
    setStage("starting");
    return controller;
  };

  const handlers = {
    onProgress: (event: ProgressEvent) => {
      const line = describe(event);
      if (line) setLines((prev) => [...prev, line]);
      setStage(stageOf(event));
    },
    onResult: setResult,
    onError: (message: string) => setError(message),
    onDone: () => {
      setRunning(false);
      setStage("");
    },
  };

  const playExample = async (example: ExampleSummary) => {
    if (running) return;
    setActiveId(example.id);
    setIsRecorded(true);
    const controller = beginStream();
    try {
      await replayExample(example.id, handlers, controller.signal);
    } catch (exc) {
      if ((exc as Error).name !== "AbortError") setError("Could not load that recording.");
      setRunning(false);
    }
  };

  const startLive = async () => {
    const trimmed = query.trim();
    if (!trimmed || running) return;
    setActiveId(null);
    setIsRecorded(false);
    const controller = beginStream();
    try {
      await runResearch(trimmed, handlers, controller.signal);
    } catch (exc) {
      if ((exc as Error).name !== "AbortError") setError("Lost connection to the server.");
      setRunning(false);
    }
  };

  const stop = () => {
    abort.current?.abort();
    setRunning(false);
    setStage("");
  };

  const tooLong = config ? query.trim().length > config.max_query_chars : false;

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Agentic Research Engine</h1>
          <p className="tagline">
            Decomposes a question, researches it in parallel, and traces every evidence-owing
            sentence back to the exact quote behind it.
          </p>
        </div>
        <a
          className="btn btn--ghost"
          href="https://github.com/NirmalKumar31/agentic-research-engine"
          target="_blank"
          rel="noreferrer noopener"
        >
          Source
        </a>
      </header>

      <section className="examples-panel">
        <div className="examples-panel__head">
          <h2>Recorded demonstrations</h2>
          <span className="badge badge--recorded">Recorded run</span>
        </div>
        <p className="muted small">
          Real executions of this engine, captured in full and replayed here. The progress
          below is the sequence the graph actually emitted — nothing is simulated. Timing is
          compressed so an eighteen-minute local run is watchable.
        </p>

        {examples.length === 0 ? (
          <p className="muted small">No recordings are available on this instance.</p>
        ) : (
          <ul className="example-list">
            {examples.map((example) => (
              <li key={example.id}>
                <button
                  type="button"
                  className={`example ${activeId === example.id ? "example--active" : ""}`}
                  onClick={() => playExample(example)}
                  disabled={running}
                >
                  <span className="example__label">{example.label}</span>
                  <span className="example__question">{example.question}</span>
                  <span className="example__facts muted small">
                    {example.sources} sources · {example.citable_evidence} citable evidence
                    items · {Math.round(example.duration_s)}s
                    {example.has_pdf_evidence && " · PDF page citations"}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>

      {liveEnabled ? (
        <section className="composer">
          <h2>Or run a live question</h2>
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question that needs several sources to answer well…"
            rows={3}
            disabled={running}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) startLive();
            }}
          />
          <div className="composer__row">
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
          </div>
          {config && (
            <p className="muted small">
              Live limits: {config.max_rounds} round, {config.max_sources} sources,{" "}
              {config.runs_per_hour} runs per hour, {config.max_runtime_seconds}s cap.
              {tooLong && (
                <strong> Question is over the {config.max_query_chars}-character limit.</strong>
              )}
            </p>
          )}
        </section>
      ) : (
        <section className="notice">
          <p className="muted small">
            Live research is disabled on this public instance. The daily run cap is held in
            process memory, and a free host that sleeps when idle resets it on every cold
            start — so it cannot bound an API quota. Rather than add a database whose only
            job is letting strangers spend the budget, the site serves recorded runs.{" "}
            <a
              href="https://github.com/NirmalKumar31/agentic-research-engine#usage"
              target="_blank"
              rel="noreferrer noopener"
            >
              Run it locally
            </a>{" "}
            for live research: the models can run on Ollama with no paid LLM
            API usage, though live web research still needs a search provider
            such as Tavily configured.
          </p>
        </section>
      )}

      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}

      {(running || lines.length > 0) && (
        <section className="progress">
          <div className="progress__head">
            <span className={`dot ${running ? "dot--live" : ""}`} />
            <strong>{running ? `Replaying — ${stage}` : "Finished"}</strong>
            {isRecorded && <span className="badge badge--recorded">Recorded run</span>}
          </div>
          <ol className="log">
            {lines.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ol>
          <div ref={logEnd} />
        </section>
      )}

      {result && (
        <>
          {isRecorded && (
            <p className="muted small recorded-note">
              This report came from a recorded run. Click any citation to open the evidence
              behind it.
            </p>
          )}
          <ReportView result={result} />
        </>
      )}

      <footer className="footer muted small">
        Evidence is verified against retrieved sources, not against the world. A confident
        report built on wrong pages will still look clean here.
      </footer>
    </div>
  );
}
