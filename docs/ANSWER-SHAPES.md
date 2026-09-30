# Answer shapes: what is decided deterministically, and what is not

The contract a run is held to comes from its answer shape. An earlier
checkpoint said "eight explicit shapes are deterministically
supported", which overstated it. This is the precise account.

## The three routes to a shape

A shape is chosen by exactly one of these, and `shape_source` on the
contract records which:

| `shape_source` | Meaning |
| --- | --- |
| `wording` | The question's wording is explicit and the model agreed with it. |
| `corrected-from-wording` | The wording is explicit and the model disagreed; the wording won. |
| `model` | The wording is **not** explicit. The model's label stands. |

## What can be corrected from wording

`question_form.shape_from_wording` returns a shape only for the forms
below. Anything else returns `None`, and `None` means "the model's
reading stands" — it is an answer, not a failure to find one.

Checked in this order, because several patterns match the same text and
the earlier reading is the more specific one:

| Shape | Fires on |
| --- | --- |
| `comparison` | two or more named subjects either side of a comparison connective (`differs from`, `versus`, `compare to`, `difference between`, `between A and B`, `how do A, B and C differ`) |
| `causal` | an auxiliary opening plus a causal verb — "does X cause Y", "did X lead to Y", "is X responsible for Y" |
| `causal_drivers` | "what causes X", "why does X happen", "what factors contribute to X", "the causes/reasons/factors of X" |
| `numeric` | "how many/much/long", "what percentage", "the size/cost/latency/… of X" |
| `procedural` | "how do I/we/you", "how to", "steps to" |
| `recommendation` | "should I/we", "which should", "is it worth" |
| `temporal` | "when was/did/will", "timeline", "history of", "latest", "current" |
| `list` | "types", "kinds", "categories", "examples", "benefits", "drawbacks", "risks", "components", "stages", "options" |
| `definition` | "what is/are X", "define X", "the meaning of X" |

**Deliberate precedence decisions**, each because two readings both
match:

- "What is the context window **size of** X?" is `numeric`, not
  `definition`. It opens like a definition and asks for a figure.
- "What is the **current** price of X?" is `numeric`, not `temporal`.
  The question wants a number; that the answer is time-sensitive is
  carried by the analysis stage's own `time_sensitive` flag rather than
  by reshaping the answer into a timeline.
- "What are the main **causes** of X?" is `causal_drivers`, not `list`.
  Its contract carries a slot for the limits of its own causal
  evidence; a bare list of members does not.
- "**Does** X cause Y?" is `causal`, not `causal_drivers`. Naming a
  plausible driver is not an answer to a yes/no causal test, and the
  two contracts share no slot by which one could stand in for the
  other.
- "What are the **steps** to reduce overfitting?" is `procedural`, not
  `list`.

## What remains model-decided

- **Any question with no explicit form.** "Tell me about vector
  databases in production", "Overview of retrieval strategies", a bare
  noun phrase. The reader returns `None` and the analyst's
  `output_format` is used unchanged.
- **`synthesis` / multi-part.** There is no wording rule for it. A
  conjunction cannot be split safely: "what are the risks **and**
  benefits of X" is one list, and "what is X **and** how much does it
  cost" is two questions. Splitting on "and" would break the first to
  fix the second.
- **`overview` as a residual.** The analyst may still choose it, and
  the prompt tells it that `overview` is the narrowest shape rather
  than a safe default.

## How multi-part questions are detected

Only by the model, via `AnalysisOut.parts`. The contract then builds one
core slot per named part, so a report answering one part of three
cannot present itself as complete.

**The wording override stands down when the model named parts.** "What
is RAG, and how much does it cost to run?" contains an explicit numeric
form, so the wording reader returns `numeric` and cannot see the
question has two halves. Overriding a model that correctly identified
both would make the answer worse, so it does not.

This is a recorded limitation: a multi-part question whose parts the
model failed to name is read as whichever single form its wording
carries.

## When the contract becomes unusable

An unusable contract **refuses every claim** — `citations/relevance.py`
returns a rejection for all of them — so a run with one publishes
nothing. It occurs in exactly two cases:

1. **A comparison with fewer than two subjects.** Deliberate: with one
   subject there is nothing to contrast, and the contract would be
   satisfiable by a definition, which is the defect it exists to stop.
2. **An `output_format` with no mapped `QuestionType`.** Now
   unreachable: a test asserts every `OutputFormat` member maps, every
   `QuestionType` is reachable, and the schema offers exactly the
   members that exist. Before that test, four of nine shapes were
   unreachable.

## When fallback occurs

| Situation | Fallback |
| --- | --- |
| Analysis call fails | A default `QueryAnalysis` is built from the raw query; the run continues and the error is recorded. |
| `output_format` is `synthesis` but no parts are named | The shape falls back to `definition` rather than producing an unusable contract, because an unusable contract silences the whole run. |
| Comparison sides cannot be parsed from wording | The analyst's `comparison_subjects` are used, then its `entities`. Refusing a run over a failed parse would be worse than the defect being fixed. |
| Query generation fails | Each sub-question's own text is used as its query, with the rationale recorded as a fallback. |

## What is *not* claimed

- That classification is deterministic in general. It is deterministic
  **for the forms listed above**, and those only.
- That the shape is always right. An explicit form read correctly can
  still be the wrong contract for what the user meant; `shape_source`
  exists so that can be audited after the fact.
