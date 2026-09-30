import { answered, questionTypeLabel, slotStatuses } from "./contract";
import type { AnswerCoverage, Contract, Report } from "./types";

/**
 * What the question was decided to require, and what the report
 * actually delivered.
 *
 * Shown because the interesting half of verification is no longer
 * whether a claim is supported but whether the report answers what
 * was asked. A run can publish three correctly cited claims and
 * answer nothing — that is a real failure the page could not
 * previously show, because every claim on it looked identical to a
 * claim that did answer.
 */
export function ContractView({
  contract,
  report,
  coverage,
}: {
  contract: Contract | null;
  report: Report;
  coverage?: AnswerCoverage | null;
}) {
  // Older recordings predate contracts entirely. Showing an empty
  // panel would imply the run was held to a contract and failed it.
  if (!contract) return null;

  if (!contract.usable) {
    return (
      <section className="contract">
        <h3>What the question requires</h3>
        <p className="notice notice--withheld">
          <strong>The question could not be given a checkable shape.</strong>{" "}
          {contract.unusable_reason} Nothing below should be read as an answer
          to it.
        </p>
      </section>
    );
  }

  const statuses = slotStatuses(contract, report);
  const isAnswered = answered(contract, report);
  const missingCore = statuses.filter((s) => s.core && !s.filled);
  // Subjects no retrieved source mentions. This is why an unfilled
  // contract may be the correct outcome rather than a failure, and it
  // is the one thing the client cannot work out for itself.
  const absent = coverage?.absent_entities ?? [];

  return (
    <section className="contract">
      <h3>What the question requires</h3>
      <p className="muted small">
        Decided before anything was retrieved, from the question alone — read as
        a <strong>{questionTypeLabel(contract)}</strong>
        {contract.entities.length > 0 && (
          <> about {contract.entities.join(" and ")}</>
        )}
        . A claim can be true, correctly cited, and still fill none of these.
      </p>

      {absent.length > 0 && (
        <p className="notice notice--withheld">
          <strong>
            No retrieved source mentions{" "}
            {absent.map((e) => `\u201c${e}\u201d`).join(", ")}.
          </strong>{" "}
          Either no reachable source covers it, or the question names something
          that does not exist. An unfilled requirement below is the correct
          outcome in that case — the engine declined to invent an answer rather
          than failing to find one.
        </p>
      )}

      <ul className="contract__slots plain-list">
        {statuses.map((slot) => (
          <li
            key={slot.name}
            className={slot.filled ? "slot slot--filled" : "slot slot--missing"}
          >
            <span className="slot__mark" aria-hidden="true">
              {slot.filled ? "✓" : "○"}
            </span>
            <span className="slot__body">
              <code>{slot.name}</code>
              {slot.core && <span className="flag flag--core">required</span>}
              <span className="muted small"> — {slot.description}</span>
              {slot.filledBy && (
                <span className="muted small">
                  {" "}
                  (filled by <code>{slot.filledBy}</code>)
                </span>
              )}
            </span>
          </li>
        ))}
      </ul>

      {isAnswered ? (
        <p className="notice notice--ok">
          <strong>Every required part was answered.</strong> Each published
          claim below is also supported by one of its own cited quotes.
        </p>
      ) : (
        <p className="notice notice--withheld">
          <strong>This report does not answer the question.</strong> Nothing
          published fills the{" "}
          <code>{missingCore.map((s) => s.name).join(" or ")}</code>{" "}
          requirement.{" "}
          {absent.length > 0
            ? "With a subject that appears in no source, that requirement could not be filled by anything true."
            : "Claims below may still be true and correctly sourced \u2014 they describe the subject rather than answering what was asked."}
        </p>
      )}
    </section>
  );
}
