import { useEffect, useRef, useState } from "react";
import { fetchConfig, runResearch, type DemoConfig } from "./api";
import { describe, stageOf } from "./progress";
import { ReportView } from "./ReportView";
import type { ProgressEvent, RunResult } from "./types";

const EXAMPLES = [
  "Compare modern approaches for detecting fraud in highly imbalanced transaction datasets.",
  "Are locally hosted open-weight language models viable for enterprise document analysis?",
  "What are the practical differences between vector databases and search engines for RAG?",
];

export default function App() {
  const [query, setQuery] = useState("");
  const [lines, setLines] = useState<string[]>([]);
  const [stage, setStage] = useState("");
  const [result, setResult] = useState<RunResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [config, setConfig] = useState<DemoConfig | null>(null);
  const abort = useRef<AbortController | null>(null);
  const logEnd = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    fetchConfig().then(setConfig);
  }, []);

  useEffect(() => {
    logEnd.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [lines]);

  // Abort an in-flight run if the component goes away, so the server can
  // release its capacity slot rather than waiting for the timeout.
  useEffect(() => () => abort.current?.abort(), []);

  const start = async () => {
    const trimmed = query.trim();
    if (!trimmed || running) return;

    setRunning(true);
    setLines([]);
    setResult(null);
    setError(null);
    setStage("starting");

    const controller = new AbortController();
    abort.current = controller;

    try {
      await runResearch(
        trimmed,
        {
          onProgress: (event: ProgressEvent) => {
            const line = describe(event);
            if (line) setLines((prev) => [...prev, line]);
            setStage(stageOf(event));
          },
          onResult: setResult,
          onError: (message) => setError(message),
          onDone: () => {
            setRunning(false);
            setStage("");
          },
        },
        controller.signal,
      );
    } catch (exc) {
      if ((exc as Error).name !== "AbortError") {
        setError("Lost connection to the server.");
      }
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
            Decomposes a question, researches it in parallel, and traces every sentence back
            to the exact quote behind it.
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

      <section className="composer">
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question that needs several sources to answer well…"
          rows={3}
          disabled={running}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) start();
          }}
        />
        <div className="composer__row">
          <div className="examples">
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                className="chip"
                disabled={running}
                onClick={() => setQuery(example)}
              >
                {example.slice(0, 46)}…
              </button>
            ))}
          </div>
          {running ? (
            <button type="button" className="btn" onClick={stop}>
              Stop
            </button>
          ) : (
            <button
              type="button"
              className="btn btn--primary"
              onClick={start}
              disabled={!query.trim() || tooLong}
            >
              Research
            </button>
          )}
        </div>
        {config && (
          <p className="muted small">
            Demo limits: {config.max_rounds} round, {config.max_sources} sources,{" "}
            {config.runs_per_hour} runs per hour, {config.max_runtime_seconds}s cap.
            {tooLong && (
              <strong> Question is over the {config.max_query_chars}-character limit.</strong>
            )}
          </p>
        )}
      </section>

      {error && (
        <div className="alert" role="alert">
          {error}
        </div>
      )}

      {(running || lines.length > 0) && (
        <section className="progress">
          <div className="progress__head">
            <span className={`dot ${running ? "dot--live" : ""}`} />
            <strong>{running ? `Working — ${stage}` : "Finished"}</strong>
          </div>
          <ol className="log">
            {lines.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ol>
          <div ref={logEnd} />
        </section>
      )}

      {result && <ReportView result={result} />}

      <footer className="footer muted small">
        Evidence is verified against retrieved sources, not against the world. A confident
        report built on wrong pages will still look clean here.
      </footer>
    </div>
  );
}
