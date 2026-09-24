import { useMemo, useState } from "react";
import { ClaimView } from "./ClaimView";
import { EvidenceDrawer } from "./EvidenceDrawer";
import type { Claim, RunResult } from "./types";

/**
 * The finished report, laid out as a document rather than a dashboard.
 *
 * One reading column; provenance and metrics live in a side rail that
 * collapses below the report on narrow screens. Wrapping every paragraph
 * in a card would make a research report look like an admin panel.
 */
/** Covered/total as the engine reported it, when a run produced one. */
export interface CoverageSummary {
  covered: number;
  total: number;
}

export function ReportView({
  result,
  coverage,
}: {
  result: RunResult;
  coverage?: CoverageSummary | null;
}) {
  const [openClaim, setOpenClaim] = useState<Claim | null>(null);

  const evidenceById = useMemo(
    () => new Map(result.evidence.map((e) => [e.id, e])),
    [result.evidence],
  );
  const sourceById = useMemo(
    () => new Map(result.sources.map((s) => [s.id, s])),
    [result.sources],
  );
  const subQuestionById = useMemo(
    () => new Map((result.plan?.sub_questions ?? []).map((q) => [q.id, q])),
    [result.plan],
  );
  // The API does not return query text, so the drawer shows the id alone
  // rather than inventing wording for it.
  const queryTextById = useMemo(() => new Map<string, string>(), []);

  const report = result.report;

  const allClaims = useMemo(() => {
    if (!report) return [];
    return [
      ...report.summary_claims,
      ...report.key_findings,
      ...report.sections.flatMap((s) => s.claims),
    ];
  }, [report]);

  const substantive = allClaims.filter((c) => c.kind !== "framing");
  const grounded = substantive.filter((c) => c.evidence_ids.length > 0);
  const citationsBySource = useMemo(() => {
    const counts = new Map<string, number>();
    for (const claim of allClaims) {
      for (const id of claim.citation_ids) counts.set(id, (counts.get(id) ?? 0) + 1);
    }
    return counts;
  }, [allClaims]);

  if (!report) return <p className="muted">No report was produced.</p>;

  const openItems = openClaim
    ? openClaim.evidence_ids
        .map((id) => evidenceById.get(id))
        .filter((e): e is NonNullable<typeof e> => Boolean(e))
    : [];

  const download = () => {
    const blob = new Blob([result.markdown], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${result.run_id}.md`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const renderClaims = (claims: Claim[]) =>
    claims.map((claim, i) => (
      <ClaimView
        key={i}
        claim={claim}
        evidenceById={evidenceById}
        selected={openClaim === claim}
        onOpen={setOpenClaim}
      />
    ));

  return (
    <div className="report-layout">
      <article className="report">
        <header className="report__head">
          <h2>{report.title}</h2>
          <p className="report__summary-line muted small">
            {substantive.length} substantive claims · {grounded.length} evidence-linked ·{" "}
            {result.sources.length} sources · {result.evidence.length} evidence items
          </p>
          <p className="hint">Select any citation to see the exact passage behind it.</p>
        </header>

        {coverage && coverage.covered < coverage.total && (
          <p className="notice notice--coverage">
            <strong>Limited evidence coverage.</strong> Research completed within the
            demo's one-round limit, but {coverage.total - coverage.covered} of{" "}
            {coverage.total} research dimensions did not reach the evidence
            threshold. The claims below are still verified against their own
            sources; the question is covered less completely than a longer run
            would manage.
          </p>
        )}

        {report.summary_claims.length > 0 && (
          <section>
            <h3>Summary</h3>
            {renderClaims(report.summary_claims)}
          </section>
        )}

        {report.key_findings.length > 0 && (
          <section>
            <h3>Key findings</h3>
            {renderClaims(report.key_findings)}
          </section>
        )}

        {report.sections.map((section, i) => (
          <section key={i}>
            <h3>{section.heading}</h3>
            {renderClaims(section.claims)}
          </section>
        ))}

        {report.contradictions.length > 0 && (
          <section>
            <h3>Disagreements between sources</h3>
            {report.contradictions.map((c, i) => (
              <div key={i} className="contradiction">
                <strong>{c.topic}</strong>
                <p>{c.left_summary}</p>
                <p>{c.right_summary}</p>
                {!c.auditable && (
                  <span className="flag flag--warn">not evidenced on both sides</span>
                )}
              </div>
            ))}
          </section>
        )}

        {report.limitations.length > 0 && (
          <section>
            <h3>Limitations</h3>
            <ul className="plain-list">
              {report.limitations.map((l, i) => (
                <li key={i}>{l}</li>
              ))}
            </ul>
          </section>
        )}

        <button type="button" onClick={download} className="btn btn--ghost btn--small">
          Download Markdown
        </button>
        <p className="report__disclaimer muted small">
          Research aid only. Verify important medical, legal, financial or other
          high-stakes decisions against authoritative primary sources.
        </p>
      </article>

      <aside className="rail">
        <section className="rail__block">
          <h4>Sources</h4>
          <ul className="source-list">
            {result.sources.map((s) => (
              <li key={s.id} className={s.usable ? "" : "source--unusable"}>
                <a href={s.url} target="_blank" rel="noreferrer noopener">
                  <code className="chip chip--id">{s.id}</code> {s.title || s.domain}
                </a>
                <span className="muted small">
                  {s.domain}
                  {s.content_origin === "pdf_extract" && " · PDF"}
                  {citationsBySource.get(s.id)
                    ? ` · ${citationsBySource.get(s.id)} citations`
                    : " · not cited"}
                </span>
              </li>
            ))}
          </ul>
        </section>

        {result.plan && result.plan.sub_questions.length > 0 && (
          <section className="rail__block">
            <h4>Research dimensions</h4>
            <ul className="dimension-list">
              {result.plan.sub_questions.map((q) => (
                <li key={q.id}>
                  <code className="chip chip--id">{q.id}</code> {q.text}
                </li>
              ))}
            </ul>
          </section>
        )}
      </aside>

      {openClaim && (
        <EvidenceDrawer
          claim={openClaim}
          items={openItems}
          sourceById={sourceById}
          subQuestionById={subQuestionById}
          queryTextById={queryTextById}
          onClose={() => setOpenClaim(null)}
        />
      )}
    </div>
  );
}
