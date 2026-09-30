# Manual review

Read by hand against the pass criteria committed **before** the run at
`docs/vnext-baseline/paid-run-acceptance.md`, so the criteria could not be
fitted to the result.

## Queries

- [x] **preserve the question's own vocabulary** — Q1 is `causes of
      overfitting in machine learning`; "overfitting" appears in all six
      queries and "machine learning" in two.
- [x] **median length within three to eight words** — median 5, min 4, max 6.
      Baseline: median 11.
- [x] **no specialist-term stacking** — no query contains a term the question
      or its sub-questions did not use.
- [x] **each query records its sub-question and a rationale** — all six do.

## Selection

- [ ] **allocated across answer slots** — *FAILED, and not at selection.*
      Selection did allocate across the sub-questions it was given candidates
      for. But the query writer issued no query for SQ4 or SQ5, so no
      candidate existed for either and no allocation could reach them. The
      manifest makes the cause unambiguous: `by_sub_question` shows
      `SQ4: 0, SQ5: 0`, and no candidate lists them.
- [x] **multi-sub-question candidates credited to all** — `www.lenovo.com`
      (SQ1, SQ3) and `www.geeksforgeeks.org` (SQ1, SQ2) each selected once
      and credited to both.
- [x] **no social or newsletter source displaces an accountable source
      within 0.35** — `www.facebook.com` (0.62), `www.linkedin.com` (0.36),
      four Reddit/StackExchange threads and eight blogs were all dropped.
      `www.udacity.com` scored 0.92 raw — higher than four selected pages —
      and was dropped once the blog penalty applied.
- [x] **every selected source and drop reason recorded** — 41 candidates, 6
      selected, 35 dropped, each with a reason. `truncated: 0`.
- [x] **fetch failure distinguishable from selection and relevance failure** —
      all six selected show `provider_content (provider_raw)`; the 35 dropped
      show `not fetched`. No fetch failures occurred this run, so that
      distinction is exercised structurally rather than in anger.

## Coverage and claims

- [x] **evidence admissible, not merely quote-exact** — 2 covered, 3 weak, 0
      missing. The three weak ones include the two unsearched sub-questions.
- [x] **every gap carries a cause** — the limitations name each one, and
      distinguish "the evidence does not establish X" from "only limited
      evidence was found for X".
- [x] **every published claim supported and relevant** — 5 published,
      `citation_integrity_rate` 1.0, `citation_coverage_rate` 1.0,
      `claim_support_rate` 0.833.
- [x] **unsupported publications: 0**
- [x] **false "answered" conclusion: 0** — `answered: true` and the core slot
      `candidate_drivers` is genuinely filled by five supported claims. The
      two optional slots are reported missing.

## Lifecycle and budget

- [x] **duplicate `started` events: 0** — exactly one visible. Baseline: two.
- [x] **within every ceiling** — 147.5s/240s, $0.008809/$0.05, 14/30 provider
      requests, 14/20 LLM calls, 25,662/120,000 input tokens,
      12,485/20,000 output tokens, 6/8 Tavily credits.
- [x] **no credentials, private paths, internal endpoints or secret-bearing
      headers** — gitleaks clean over the directory; zero matches for
      `api_key`, `Authorization`, `Bearer`, `sk-`, `tvly-`, `/Users/`,
      `onrender` in the raw stream; all 41 manifest URLs free of query
      strings, fragments and userinfo.

## Verdict

**Pass, with one failure that the run itself made visible.**

The release did what it was built to do: better queries, better sources, a
correct answer shape, honest coverage, five supported claims where the
baseline published none, and a manifest that preserves what previously
vanished.

The failure — two sub-questions never searched — is a pre-existing defect in
query budgeting that only became observable because of this release's
observability work. Reporting it is the point. A run that hid it would have
looked better and been worth less.
