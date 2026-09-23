import { useState } from "react";
import type { Claim, Evidence, Source } from "./types";

interface Props {
  claim: Claim;
  evidenceById: Map<string, Evidence>;
  sourceById: Map<string, Source>;
}

/**
 * A claim with its citations, expandable to the exact evidence behind it.
 *
 * This is the point of the whole provenance redesign: the reader can open
 * any sentence and see the verbatim span, the page it came from and the
 * source, rather than being asked to trust a bracketed number.
 */
export function ClaimView({ claim, evidenceById, sourceById }: Props) {
  const [open, setOpen] = useState(false);
  const items = claim.evidence_ids
    .map((id) => evidenceById.get(id))
    .filter((e): e is Evidence => Boolean(e));

  const markers = claim.citation_ids.map((id) => {
    const page = items.find((e) => e.source_id === id && e.page)?.page;
    return page ? `[${id}, p. ${page}]` : `[${id}]`;
  });

  return (
    <div className={`claim claim--${claim.kind}`}>
      <p className="claim__text">
        {claim.text}
        {markers.length > 0 && (
          <button
            type="button"
            className="claim__markers"
            onClick={() => setOpen(!open)}
            aria-expanded={open}
            title="Show the evidence behind this claim"
          >
            {markers.join("")}
          </button>
        )}
        {claim.kind === "synthesis" && (
          <span className="tag tag--synthesis" title="Drawn across several sources">
            synthesis
          </span>
        )}
        {claim.kind !== "framing" && claim.evidence_ids.length === 0 && (
          <span className="tag tag--warn" title="This claim carries no evidence">
            uncited
          </span>
        )}
      </p>

      {open && (
        <div className="evidence-panel">
          {items.length === 0 && <p className="muted">No evidence resolved for this claim.</p>}
          {items.map((item) => {
            const source = sourceById.get(item.source_id);
            return (
              <div key={item.id} className="evidence">
                <div className="evidence__head">
                  <code>{item.id}</code>
                  <span className={`tag tag--${item.stance}`}>{item.stance}</span>
                  {item.page && <span className="tag">page {item.page}</span>}
                  <span
                    className={`tag ${item.citable ? "tag--ok" : "tag--warn"}`}
                    title={
                      item.citable
                        ? "Quote found verbatim in the source"
                        : "Quote could not be matched exactly"
                    }
                  >
                    {item.quote_match.replace("_", " ")}
                  </span>
                </div>
                <blockquote className="evidence__quote">{item.quote}</blockquote>
                <p className="evidence__meta">
                  {item.claim}
                  {source && (
                    <>
                      {" — "}
                      <a href={source.url} target="_blank" rel="noreferrer noopener">
                        {source.title || source.domain}
                      </a>{" "}
                      <span className="muted">
                        ({source.domain}, {source.content_origin.replace("_", " ")})
                      </span>
                    </>
                  )}
                </p>
                <p className="evidence__trace muted">
                  answers {item.sub_question_id}
                  {item.query_id
                    ? ` · found by ${item.query_id}`
                    : " · noticed while researching another sub-question"}
                </p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
