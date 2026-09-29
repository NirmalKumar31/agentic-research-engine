import type { Claim, Contract, Report } from "./types";

/**
 * Which parts of the answer the published report actually filled.
 *
 * The interesting half of verification is no longer "is this claim
 * supported" but "does the report answer what was asked". A run can
 * publish three correctly cited claims and answer nothing, which is
 * what happened before the contract existed, so the page has to be
 * able to show the difference.
 *
 * Derived from the published claims rather than read from a field.
 * The engine computes the same thing for the report's limitations,
 * and a second copy shipped to the client could disagree with it —
 * this way both are functions of what was published.
 */

export interface SlotStatus {
  name: string;
  description: string;
  core: boolean;
  filled: boolean;
  /**
   * The slot that actually discharged this one, when an alternative
   * did. A comparison is answered by a contrast *or* by a
   * relationship, and a reader should see which arrived.
   */
  filledBy: string | null;
}

export function publishedClaims(report: Report): Claim[] {
  return [
    ...report.summary_claims,
    ...report.key_findings,
    ...report.sections.flatMap((s) => s.claims),
  ];
}

/** Slots the published claims declared. Blank declarations are dropped:
 * a claim that named no slot fills none. */
export function declaredSlots(report: Report): string[] {
  return publishedClaims(report)
    .map((c) => c.answer_slot ?? "")
    .filter((slot) => slot.length > 0);
}

export function slotStatuses(contract: Contract, report: Report): SlotStatus[] {
  const declared = new Set(declaredSlots(report));
  return contract.required_slots.map((slot) => {
    if (declared.has(slot.name)) {
      return { ...slot, filled: true, filledBy: null };
    }
    const alternative = (slot.satisfied_by ?? []).find((name) => declared.has(name));
    return {
      name: slot.name,
      description: slot.description,
      core: slot.core,
      filled: alternative !== undefined,
      filledBy: alternative ?? null,
    };
  });
}

/**
 * True only when every core slot is discharged.
 *
 * Not "at least one", and not "some claims published". A comparison
 * that published two definitions has answered nothing, and a
 * multi-part question answered in one part of three is not answered.
 */
export function answered(contract: Contract, report: Report): boolean {
  const statuses = slotStatuses(contract, report);
  const core = statuses.filter((s) => s.core);
  return core.length > 0 && core.every((s) => s.filled);
}

/** Human label for the question shape. */
export function questionTypeLabel(contract: Contract): string {
  return contract.question_type.replace(/_/g, " ");
}
