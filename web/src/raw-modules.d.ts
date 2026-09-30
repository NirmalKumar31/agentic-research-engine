/**
 * Vite's `?raw` suffix returns a module's own source as a string.
 *
 * Declared here because this project has no `@types/node`, and the
 * wiring tests need to read source text rather than mount components:
 * there is no DOM renderer, so what a component *renders* cannot be
 * asserted, only what it is wired to. `?raw` keeps that possible
 * without adding a dependency whose only use would be in tests.
 */
declare module "*?raw" {
  const content: string;
  export default content;
}
