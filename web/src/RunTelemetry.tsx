/**
 * The run's own instrumentation, for a reader who wants to check it.
 *
 * The headline numbers are always visible; the sixty-odd rows behind
 * them are not. The panel first shipped entirely collapsed and the very
 * next thing asked of it was for the numbers it was already carrying,
 * which is a discoverability problem rather than a missing feature.
 *
 * Every value comes from `telemetry.ts`, which derives nothing the
 * server did not send.
 */

import {
  formatSeconds,
  headlineStats,
  stageBreakdown,
  stageLabel,
  telemetryGroups,
} from "./telemetry";
import type { Metrics } from "./types";

export function RunTelemetry({ metrics }: { metrics: Metrics }) {
  const groups = telemetryGroups(metrics);
  const stages = stageBreakdown(metrics);
  if (groups.length === 0) return null;

  return (
    <section className="telemetry" aria-labelledby="telemetry-heading">
      <div className="telemetry__headline">
        <h3 id="telemetry-heading">Run telemetry</h3>
        <p className="muted small">
          Measured by the engine during this run, not estimated afterwards. No
          credential is recorded or shown.
        </p>
        <ul className="telemetry__stats">
          {headlineStats(metrics).map((stat) => (
            <li key={stat.label}>
              <span className="telemetry__stat-value">{stat.value}</span>
              <span className="telemetry__stat-label muted small">
                {stat.label}
                {stat.note && ` (${stat.note})`}
              </span>
            </li>
          ))}
        </ul>
      </div>

      <details className="telemetry__detail">
        <summary className="telemetry__summary">
          Full breakdown — timing per stage, calls, tokens, retrieval,
          verification, budgets and build
        </summary>

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
    </section>
  );
}
