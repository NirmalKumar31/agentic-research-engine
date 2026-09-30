import { describe, expect, it } from "vitest";
import contractView from "./ContractView.tsx?raw";
import reportView from "./ReportView.tsx?raw";
import types from "./types.ts?raw";

/**
 * That the absent-subject notice is reachable.
 *
 * There is no DOM renderer in this project, so the component cannot be
 * mounted and asserted against. What *can* be checked is the wiring,
 * and the wiring is the part that has repeatedly been wrong: this
 * repository has produced six defects where a value was computed
 * correctly and then never passed to the thing that needed it, and
 * four of them passed every test because the fakes constructed the
 * object correctly while production did not.
 *
 * `absent_entities` is specifically at risk because it is the one
 * coverage fact the client cannot derive for itself -- slot status is
 * computed from the published claims in `contract.ts`, but whether a
 * subject appeared in any retrieved source is a fact about source text
 * the browser never receives. If it is not handed down explicitly, the
 * panel silently goes back to reporting "0 of 5 requirements covered"
 * for a question that named something that does not exist.
 */
describe("the absent-subject notice is wired, not merely defined", () => {
  it("ReportView hands the coverage down to the panel", () => {
    expect(reportView).toMatch(/coverage=\{result\.answer_coverage\}/);
  });

  it("the panel reads absent_entities and renders it", () => {
    expect(contractView).toContain("absent_entities");
    expect(contractView).toContain("No retrieved source mentions");
  });

  it("the notice appears before the slot list, because it is the reason", () => {
    expect(contractView.indexOf("No retrieved source mentions")).toBeLessThan(
      contractView.indexOf("contract__slots"),
    );
  });

  it("the result type carries the field the server sends", () => {
    expect(types).toMatch(/answer_coverage: AnswerCoverage \| null/);
  });
});
