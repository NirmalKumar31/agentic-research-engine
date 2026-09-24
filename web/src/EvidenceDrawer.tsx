import { useEffect, useRef } from "react";
import type { Claim, Evidence, Source, SubQuestion } from "./types";

interface Props {
  claim: Claim;
  items: Evidence[];
  sourceById: Map<string, Source>;
  subQuestionById: Map<string, SubQuestion>;
  queryTextById: Map<string, string>;
  onClose: () => void;
}

/**
 * The evidence behind one claim.
 *
 * This panel is the product. Everything else on the page exists so that
 * clicking a citation can show the verbatim span, the page it came from
 * and the path that retrieved it -- or say plainly that the last of those
 * does not exist, which is the honest answer for cross-attributed items.
 */
export function EvidenceDrawer({
  claim,
  items,
  sourceById,
  subQuestionById,
  queryTextById,
  onClose,
}: Props) {
  const panel = useRef<HTMLDivElement | null>(null);

  // Escape closes, and focus moves into the panel so keyboard users are
  // not left behind on the citation they just activated.
  useEffect(() => {
    panel.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <>
      <div className="drawer__scrim" onClick={onClose} aria-hidden="true" />
      <aside
        className="drawer"
        ref={panel}
        tabIndex={-1}
        role="dialog"
        aria-modal="true"
        aria-label="Evidence behind this claim"
      >
        <header className="drawer__head">
          <span className="drawer__eyebrow">
            {items.every((i) => i.citable) ? "Exact-match evidence" : "Evidence"}
          </span>
          <button type="button" className="drawer__close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>

        <p className="drawer__claim">{claim.text}</p>

        {items.length === 0 && (
          <p className="muted">No evidence resolved for this claim. It carries no citation.</p>
        )}

        {items.map((item) => {
          const source = sourceById.get(item.source_id);
          const subQuestion = subQuestionById.get(item.sub_question_id);
          const isPdf = source?.content_origin === "pdf_extract";
          return (
            <article key={item.id} className="ev">
              <div className="ev__chips">
                <code className="chip chip--id">{item.id}</code>
                {isPdf && item.page && (
                  <span className="chip chip--pdf">
                    PDF <strong>p. {item.page}</strong>
                  </span>
                )}
                <span className={`chip ${item.citable ? "chip--ok" : "chip--warn"}`}>
                  {item.quote_match === "exact_normalized"
                    ? "Exact match"
                    : item.quote_match === "fuzzy"
                      ? "Reworded — not citable"
                      : "Unmatched — not citable"}
                </span>
                {item.stance !== "neutral" && (
                  <span className="chip">{item.stance}</span>
                )}
              </div>

              <blockquote className="ev__quote">{item.quote}</blockquote>

              <dl className="ev__meta">
                {source && (
                  <>
                    <dt>Source</dt>
                    <dd>
                      <a href={source.url} target="_blank" rel="noreferrer noopener">
                        {source.title || source.domain} <span aria-hidden="true">↗</span>
                      </a>
                      <span className="muted"> · {source.domain}</span>
                    </dd>
                  </>
                )}
                {subQuestion && (
                  <>
                    <dt>Answers</dt>
                    <dd>
                      <code>{item.sub_question_id}</code> {subQuestion.text}
                    </dd>
                  </>
                )}
                {item.query_id ? (
                  <>
                    <dt>Found by</dt>
                    <dd>
                      <code>{item.query_id}</code>{" "}
                      {queryTextById.get(item.query_id) ?? ""}
                    </dd>
                  </>
                ) : (
                  <>
                    <dt>Discovery</dt>
                    <dd>
                      <span className="chip chip--muted">Cross-attributed</span>
                      <span className="ev__note muted">
                        This source was retrieved for a different research question and
                        happened to contain evidence for this claim. No query is shown
                        because none genuinely retrieved it for this question.
                      </span>
                    </dd>
                  </>
                )}
              </dl>
            </article>
          );
        })}
      </aside>
    </>
  );
}
