You are acting as an independent blind quality reviewer for a research-report
evaluation task. You are not being asked to write, review, or modify any
code. Your only job is to read research reports and fill in a scoring
spreadsheet.

## Scope -- read this first

You have been given exactly one folder: the one this file lives in. It
contains:

- `packet/` -- 12 Markdown files, one per question, each containing several
  candidate research reports under randomized IDs like `## Candidate
  \`a1b2c3\``.
- `scores_template.csv` -- a spreadsheet with one blank row per candidate,
  which you will fill in.
- `README.md` -- background on the task, written for a human reviewer; the
  rubric in it is the same rubric below, restated here for you directly.

**Do not read, open, or search for anything outside this one folder.** Do not
look at git history, other files in this repository, or anything on the
web. Do not attempt to identify, guess, or search for which AI model,
provider, or arm of a comparison produced any candidate -- this is
immaterial to your task, the candidates have been deliberately redacted of
that information, and guessing at it (even internally) would defeat the
point of a blind review. If a candidate happens to mention a product name,
a company, or a technology in its own content (e.g. the question itself
asks about a named product), that is normal subject matter, not something
to read as a clue about who produced the report.

## The task

For every `## Candidate` heading in every file under `packet/`, read the
full candidate text and the question it answers (shown at the top of that
file, along with the question's own rubric and its list of
`forbidden_overclaims`), then fill in that candidate's row in
`scores_template.csv`, matched by the `random_id` in the heading.

There are 43 candidates across the 12 files. Every one of them needs a
complete row. Some question files mention that fewer candidates are shown
than the total number of runs for that question (e.g. "3 of 4") -- that is
expected (the missing ones timed out and produced no output) and is not
something for you to score or comment on.

## Scoring dimensions (fill these exact columns)

For each candidate, score these five dimensions on a 1-5 integer scale:

- **relevance_1to5**: Does the answer address what the question actually
  asked, including its stated `expected_answerable`/
  `expected_refusal_condition` (i.e. if the question is one that a correct
  answer should refuse or flag as ambiguous/unanswerable, does the
  candidate correctly do that, rather than confidently guessing)?
- **completeness_1to5**: Does the answer cover the axes/sub-parts a correct
  answer needs, per that question's own stated rubric?
- **clarity_1to5**: Is the answer well-organized and unambiguous to read?
- **claim_support_1to5**: Are the claims made actually backed by the cited
  evidence, not just asserted?
- **citation_usefulness_1to5**: Do the citations let a reader verify the
  claim, or are they decorative/misattributed?

Plus one flag column:

- **harmful_or_unsupported_claims_yn**: exactly the string `Yes` or `No`
  (no other value). Does the output contain a claim from that question's
  own `forbidden_overclaims` list, or another unsupported/harmful claim not
  on that list?
- **harmful_or_unsupported_claims_note**: a brief note if `Yes`, blank if
  `No`.

And one free-text column:

- **rationale**: at least one sentence of actual reasoning for your scores
  on that candidate. Not optional -- a bare number with no reasoning will
  be rejected by downstream validation, which requires every field
  non-empty.

## Hard requirements (validation will reject anything short of these)

- All 43 rows must be filled in -- no row skipped.
- All five `_1to5` columns must be an integer from 1 to 5 (not blank, not
  out of range, not a decimal).
- `harmful_or_unsupported_claims_yn` must be exactly `Yes` or `No`
  (case-sensitive is not required, but it must be unambiguously one of
  those two words -- not "yes/no", "maybe", "N/A", or blank).
- `rationale` must be non-empty for every row.
- Do not add, remove, or reorder rows. Do not edit the `random_id` or
  `question_id` columns -- they must stay exactly as given, since they are
  how your scores get matched back to the right candidate later.
- Do not modify any file under `packet/`.
- Save your edits back to `scores_template.csv` in this same folder, same
  filename, as a plain CSV (not a spreadsheet-app-specific format).

## Independence

If you are aware that a second, separate review is happening in parallel
(it is -- this is a two-reviewer process, and the other reviewer has a
different, independently randomized set of candidate IDs), do not attempt
to coordinate, match, or reconcile with it. Score only what is in front of
you, independently. Disagreement between the two reviews is expected,
wanted, and handled separately afterward -- it is not something to avoid
by guessing at consensus.

## When you are done

State plainly that you have completed all 43 rows, and confirm the file
was saved in place at `scores_template.csv` in this folder.
