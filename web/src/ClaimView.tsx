import type { Claim, Evidence } from "./types";

interface Props {
  claim: Claim;
  evidenceById: Map<string, Evidence>;
  selected: boolean;
  onOpen: (claim: Claim) => void;
}

/**
 * One claim in the report, with its citation markers.
 *
 * The markers are buttons rather than decoration: activating one opens the
 * evidence drawer. They work from the keyboard, because a citation you can
 * only reach with a mouse is not a citation a reader can check.
 */
export function ClaimView({ claim, evidenceById, selected, onOpen }: Props) {
  const items = claim.evidence_ids
    .map((id) => evidenceById.get(id))
    .filter((e): e is Evidence => Boolean(e));

  // [S4] normally, [S4, p. 5] when that source gave us a page. The page
  // comes from the evidence, never from a guess about the document.
  const markers = claim.citation_ids.map((id) => {
    const page = items.find((e) => e.source_id === id && e.page)?.page;
    return { id, text: page ? `${id}, p. ${page}` : id, page };
  });

  const uncited = claim.kind !== "framing" && claim.evidence_ids.length === 0;

  return (
    <p className={`claim ${selected ? "claim--selected" : ""} ${uncited ? "claim--uncited" : ""}`}>
      <span className="claim__text">{claim.text}</span>
      {markers.map((m) => (
        <button
          key={m.id}
          type="button"
          className={`cite ${m.page ? "cite--paged" : ""}`}
          onClick={() => onOpen(claim)}
          aria-label={`Show evidence for this claim from source ${m.text}`}
        >
          [{m.text}]
        </button>
      ))}
      {uncited && (
        <span className="flag flag--warn" title="Owes evidence and carries none">
          uncited
        </span>
      )}
    </p>
  );
}
