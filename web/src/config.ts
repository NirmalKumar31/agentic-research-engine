import type { DemoConfig } from "./api";

/**
 * Demo limits described from the server's own numbers.
 *
 * "1 research round" was hard-coded in the page while the round limit
 * lived in server configuration. The two could not disagree only
 * because the limit happened to be 1: any change would have left the
 * page advertising a bound the engine no longer had, and a visitor has
 * no way to check. Copy that states a limit has to read it.
 */
export function roundsLabel(config: DemoConfig | null): string {
  const rounds = config?.max_rounds;
  if (!rounds || rounds < 1) {
    // Absent rather than assumed. Naming a number we do not have is
    // the defect being fixed, so say what is true instead.
    return "bounded research";
  }
  return rounds === 1 ? "1 research round" : `up to ${rounds} research rounds`;
}
