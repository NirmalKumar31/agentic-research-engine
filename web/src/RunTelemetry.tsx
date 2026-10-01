/**
 * The run's own instrumentation, for a reader who wants to check it.
 *
 * Collapsed by default: it is the answer to "what happened behind this",
 * not part of the answer to the question, and putting it inline would
 * bury the report. Every number comes from `telemetry.ts`, which derives
 * nothing the server did not send.
 */

import { stageBreakdown, stageLabel, formatSeconds, telemetryGroups } from "./telemetry";
import type { Metrics } from "./types";

export function RunTelemetry({ metrics }: { metrics: Metrics }) {
  const groups = telemetryGroups(metrics);
  const stages = stageBreakdown(metrics);
  if (groups.length === 0) return null;

  return (
    <details className="telemetry">
      <summary className="telemetry__summary">
        <span>Run telemetry</span>
        <span className="muted small">
          {formatSeconds(metrics.duration_s)} · {metrics.llm_calls} model calls ·{" "}
          {(metrics.input_tokens + metrics.output_tokens).toLocaleString("en-US")} tokens
        </span>
      </summary>

      <p className="telemetry__preamble muted small">
        Measured by the engine during this run, not estimated afterwards. No
        credential is recorded or shown: the server builds this from an
        explicit allowlist and reads no environment variables.
      </p>

      {stages.length > 0 && (
        <section className="telemetry__group">
          <h4>Time per stage</h4>
          <ul className="stage-bars">
            {stages.map((slice) => (
              <li key={slice.stage} className="stage-bar">
                <span className="stage-bar__label">{stageLabel(slice.stage)}</span>
                <span className="stage-bar__track" aria-hidden="true">
                  <span
                    className="stage-bar__fill"
                    style={{ width: `${Math.max(slice.share * 100, 0.5)}%` }}
                  />
                </span>
                <span className="stage-bar__value">{formatSeconds(slice.seconds)}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="telemetry__grid">
        {groups.map((group) => (
          <section key={group.title} className="telemetry__group">
            <h4>{group.title}</h4>
            {group.blurb && <p className="muted small">{group.blurb}</p>}
            <dl className="telemetry__rows">
              {group.rows.map((r) => (
                <div key={r.label} className="telemetry__row">
                  <dt>{r.label}</dt>
                  <dd>
                    {r.value}
                    {r.note && <span className="muted small telemetry__note">{r.note}</span>}
                  </dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>
    </details>
  );
}
