import { ClaimView } from "./ClaimView";
import type { RunResult } from "./types";

function percent(value: number | null | undefined): string {
  return value === null || value === undefined ? "n/a" : `${Math.round(value * 100)}%`;
}

export function ReportView({ result }: { result: RunResult }) {
  const { report, metrics, verification, plan } = result;
  const evidenceById = new Map(result.evidence.map((e) => [e.id, e]));
  const sourceById = new Map(result.sources.map((s) => [s.id, s]));
  const cited = new Set(
    result.report
      ? [
          ...result.report.summary_claims,
          ...result.report.key_findings,
          ...result.report.sections.flatMap((s) => s.claims),
        ].flatMap((c) => c.citation_ids)
      : [],
  );

  if (!report) return <p className="muted">No report was produced.</p>;

  const download = () => {
    const blob = new Blob([result.markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${result.run_id}.md`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const support = metrics.support_breakdown ?? {};

  return (
    <div className="report">
      <div className="report__head">
        <h2>{report.title}</h2>
        <button type="button" onClick={download} className="btn btn--ghost">
          Download Markdown
        </button>
      </div>

      <p className="hint">
        Click any citation marker to see the exact quote behind that sentence.
      </p>

      {report.summary_claims.length > 0 && (
        <section>
          <h3>Summary</h3>
          {report.summary_claims.map((claim, i) => (
            <ClaimView key={i} claim={claim} evidenceById={evidenceById} sourceById={sourceById} />
          ))}
        </section>
      )}

      {report.key_findings.length > 0 && (
        <section>
          <h3>Key findings</h3>
          {report.key_findings.map((claim, i) => (
            <ClaimView key={i} claim={claim} evidenceById={evidenceById} sourceById={sourceById} />
          ))}
        </section>
      )}

      {report.sections.map((section, i) => (
        <section key={i}>
          <h3>{section.heading}</h3>
          {section.claims.map((claim, j) => (
            <ClaimView key={j} claim={claim} evidenceById={evidenceById} sourceById={sourceById} />
          ))}
        </section>
      ))}

      {report.contradictions.length > 0 && (
        <section>
          <h3>Where sources disagree</h3>
          <p className="muted">Preserved rather than resolved; both sides are traceable.</p>
          {report.contradictions.map((c, i) => (
            <div key={i} className="contradiction">
              <strong>{c.topic}</strong>
              {!c.auditable && <span className="tag tag--warn">one side unevidenced</span>}
              <p>
                {c.left_summary} {c.left_citation_ids.map((id) => `[${id}]`).join("")}
              </p>
              <p className="muted">versus</p>
              <p>
                {c.right_summary} {c.right_citation_ids.map((id) => `[${id}]`).join("")}
              </p>
            </div>
          ))}
        </section>
      )}

      {report.limitations.length > 0 && (
        <section>
          <h3>Limitations</h3>
          <ul>
            {report.limitations.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        </section>
      )}

      {plan && plan.sub_questions.length > 0 && (
        <details className="panel">
          <summary>Research plan ({plan.sub_questions.length} sub-questions)</summary>
          <ul>
            {plan.sub_questions.map((q) => (
              <li key={q.id}>
                <code>{q.id}</code> {q.text}
                {q.is_followup && <span className="tag">follow-up</span>}
                <div className="muted">{q.rationale}</div>
              </li>
            ))}
          </ul>
        </details>
      )}

      <section>
        <h3>Sources</h3>
        <ol className="sources">
          {result.sources.map((source) => (
            <li key={source.id} className={source.usable ? "" : "source--unusable"}>
              <code>{source.id}</code>{" "}
              <a href={source.url} target="_blank" rel="noreferrer noopener">
                {source.title || source.url}
              </a>
              <div className="muted">
                {source.domain} · {source.source_type} · via{" "}
                {source.content_origin.replace("_", " ")}
                {source.page_count ? ` · ${source.page_count} pages` : ""} · quality{" "}
                {source.quality_score.toFixed(2)}
                {!source.usable && ` · unusable (${source.fetch_status})`}
                {source.usable && !cited.has(source.id) && " · retrieved, not cited"}
              </div>
            </li>
          ))}
        </ol>
      </section>

      <section>
        <h3>Verification</h3>
        <table className="metrics">
          <tbody>
            <tr>
              <td>Evidence references resolved</td>
              <td>
                {percent(metrics.evidence_integrity_rate)}{" "}
                <span className="muted">
                  ({verification?.resolvable_evidence_refs ?? 0} of{" "}
                  {verification?.total_evidence_refs ?? 0})
                </span>
              </td>
            </tr>
            <tr>
              <td>Evidence-owing claims cited</td>
              <td>{percent(metrics.citation_coverage_rate)}</td>
            </tr>
            <tr>
              <td>
                Claim support{" "}
                <span className="muted">
                  ({metrics.entailment_exhaustive ? "every claim" : "sampled"})
                </span>
              </td>
              <td>
                {support.supported ?? 0} supported · {support.partially_supported ?? 0} partial
                · {support.unsupported ?? 0} unsupported · {support.not_checked ?? 0} unchecked
              </td>
            </tr>
            <tr>
              <td>Quotes verbatim</td>
              <td>
                {metrics.exact_quotes} exact · {metrics.fuzzy_quotes} fuzzy (not citable) ·{" "}
                {metrics.unmatched_quotes} unmatched
              </td>
            </tr>
          </tbody>
        </table>
        <p className="muted small">
          Verification measures faithfulness to retrieved sources, not whether the sources are
          correct about the world.
        </p>
      </section>

      <section>
        <h3>Run</h3>
        <table className="metrics">
          <tbody>
            <tr>
              <td>Mode</td>
              <td>{metrics.mode}</td>
            </tr>
            <tr>
              <td>Models</td>
              <td>
                {Object.entries(metrics.model_assignments ?? {})
                  .map(([role, spec]) => `${role}=${spec}`)
                  .join(", ") || "n/a"}
              </td>
            </tr>
            <tr>
              <td>Rounds / queries / sources</td>
              <td>
                {metrics.research_rounds} / {metrics.search_queries} / {metrics.unique_sources}{" "}
                ({metrics.distinct_domains} domains)
              </td>
            </tr>
            <tr>
              <td>Model calls / tokens</td>
              <td>
                {metrics.llm_calls} · {metrics.input_tokens.toLocaleString()} in /{" "}
                {metrics.output_tokens.toLocaleString()} out
              </td>
            </tr>
            <tr>
              <td>Estimated cost</td>
              <td>
                ${metrics.known_cost_usd.toFixed(4)}
                {!metrics.cost_is_complete && " (some models unpriced)"}
              </td>
            </tr>
            <tr>
              <td>Duration</td>
              <td>{metrics.duration_s.toFixed(1)}s</td>
            </tr>
            <tr>
              <td>Stopped because</td>
              <td>{metrics.stop_reason || "coverage sufficient"}</td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  );
}
