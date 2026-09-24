import { STAGES, type FanOutQuery, type PipelineState } from "./pipelineState";

/**
 * The research pipeline, drawn from real event state.
 *
 * A stage only lights up once the engine has emitted an event belonging to
 * it. Nothing advances on a timer, so a run that stalls at Search shows a
 * pipeline stalled at Search.
 */
export function Pipeline({ state, live }: { state: PipelineState; live: boolean }) {
  const searchActive = state.states.search === "active" && state.queries.length > 0;

  return (
    <div className="pipeline" role="group" aria-label="Research pipeline progress">
      <ol className="pipeline__track">
        {STAGES.map((stage, i) => {
          const status = state.states[stage.id];
          const note = stageNote(stage.id, state);
          return (
            <li key={stage.id} className={`stage stage--${status}`} title={note ?? undefined}>
              <span className="stage__dot" aria-hidden="true">
                {status === "done" ? "✓" : status === "warning" ? "!" : status === "failed" ? "✕" : i + 1}
              </span>
              <span className="stage__label">{stage.label}</span>
              {note && <span className="stage__note muted">{note}</span>}
              {i < STAGES.length - 1 && <span className="stage__link" aria-hidden="true" />}
              <span className="sr-only">
                {stage.label}: {status}
                {note ? `. ${note}` : ""}
              </span>
            </li>
          );
        })}
      </ol>

      {searchActive && <FanOut queries={state.queries} sources={state.counts.sources} />}

      {!searchActive && state.queries.length > 0 && (
        <p className="pipeline__collapsed muted small">
          {state.queries.length} {state.queries.length === 1 ? "search" : "searches"}
          {state.counts.sources !== null && ` · ${state.counts.sources} sources`}
        </p>
      )}

      <Counters state={state} live={live} />
    </div>
  );
}

/**
 * Short line under a stage that lost something but carried on.
 *
 * Says what survived, not just that something broke: "4/6 usable" is the
 * fact a reader needs, where a bare warning icon reads as "retrieval
 * failed" for a stage that did produce the evidence behind the report.
 */
function stageNote(id: string, state: PipelineState): string | null {
  const failures = state.issues[id] ?? 0;
  if (!failures) return null;
  if (id === "retrieve") {
    const { sources, sourcesAttempted } = state.counts;
    if (sources !== null && sourcesAttempted !== null) {
      const noun = failures === 1 ? "source" : "sources";
      return `${sources}/${sourcesAttempted} usable · ${failures} ${noun} could not be fetched`;
    }
  }
  const noun = failures === 1 ? "item" : "items";
  return `${failures} ${noun} failed, research continued`;
}

/** The fan-out, shown only while searches are in flight. */
function FanOut({ queries, sources }: { queries: FanOutQuery[]; sources: number | null }) {
  return (
    <div className="fanout" aria-label="Parallel searches">
      <div className="fanout__queries">
        {queries.map((q) => (
          <div key={q.id} className="fanout__query">
            <code className="fanout__id">{q.id}</code>
            <span className="fanout__text" title={q.text}>
              {q.text || "…"}
            </span>
            <span className="fanout__count">
              {q.results === null ? "failed" : `${q.results} results`}
            </span>
          </div>
        ))}
      </div>
      {sources !== null && <p className="muted small">{sources} sources kept after deduplication</p>}
    </div>
  );
}

/** Counts the engine reported. Anything unknown is simply absent. */
function Counters({ state, live }: { state: PipelineState; live: boolean }) {
  const items: [string, string][] = [];
  const c = state.counts;
  if (c.subQuestions !== null) items.push([String(c.subQuestions), "questions"]);
  if (c.sources !== null) items.push([String(c.sources), "sources"]);
  if (c.evidence !== null) items.push([String(c.evidence), "evidence"]);
  if (c.citations !== null) items.push([String(c.citations), "citations"]);
  if (c.coverage) items.push([`${c.coverage.covered}/${c.coverage.total}`, "covered"]);
  if (items.length === 0) return null;

  return (
    <dl className={`counters ${live ? "counters--live" : ""}`}>
      {items.map(([value, label]) => (
        <div key={label} className="counter">
          <dt className="counter__value">{value}</dt>
          <dd className="counter__label">{label}</dd>
        </div>
      ))}
    </dl>
  );
}
