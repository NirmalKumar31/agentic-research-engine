# What empirical research measures and finds about AI coding assistants and developer productivity

**Question:** What is the measured evidence on whether AI coding assistants improve developer productivity?

## What studies measure

One transcript-based study compared coding-agent transcripts with hypothetical baseline task times estimated using LLM-as-a-judge methods. [S5]

## Controlled experiments on speed and task completion

A METR study reported that AI assistants slowed experienced open-source developers on large, complex projects by as much as 20%. [S5]

## Quality, review, and net productivity

The study’s panel GMM models found that accumulated technical debt subsequently reduced future velocity. [S8]

One study focused on task-completion time rather than code quality. [S10]

## Variation by task and developer

That source identified developer over-optimism, low AI reliability, and high task complexity as potential slowdown mechanisms. [S8]

## Limitations

- The supplied evidence does not establish how field studies account for concurrent changes in staffing, workload, or development processes when estimating assistant effects.
- The supplied evidence does not establish end-to-end time differences for the same tasks after including review, verification, testing, rework, and defect resolution.
- The supplied evidence does not establish the methods or estimates behind research on security vulnerabilities in AI-generated code.
- The supplied evidence offers only limited detail about task difficulty, participant experience distributions, outcome definitions, and confidence intervals across controlled studies.
- The supplied evidence does not establish a general net productivity effect after accounting for code quality, review effort, rework, and verification.
- The retrieved evidence did not answer: Among published field studies of AI coding assistants, which studies use a matched control group, staggered rollout, or other credible counterfactual, and how do they account for concurrent changes in staffing, workload, or development process.
- The retrieved evidence did not answer: In studies that measure coding speed or output, which also record time spent on review, verification, testing, rework, and defect resolution for the same tasks, and what is the resulting end-to-end time difference.
- The retrieved evidence did not answer: In primary studies that report AI-generated-code security vulnerabilities, what vulnerability definition, evaluation procedure, sample size, and comparison baseline produced the estimate.
- Only limited evidence was found for: For controlled experiments reporting different speed effects, what were the exact task types and difficulty levels, participant experience distributions, outcome definitions, and confidence intervals in each primary study.
- Only limited evidence was found for: How do assistants affect code quality, defects, rework, and review or verification effort, and how do these outcomes change the net productivity estimate.
- Only limited evidence was found for: How do effects vary by task type and complexity, developer experience, tool, and workflow—and under what conditions do studies find little or negative productivity impact.
- 1 extracted finding(s) were excluded because their quotes could not be located in the source text
- 35 generated claim(s) were excluded because the cited evidence did not support them, or because verification did not reach them within this run's budget.
- 2 reported disagreement(s) were excluded because the evidence did not support both sides as stated.

## Sources

- **[S1]** [AI Coding Productivity — What 2026 Studies Show](https://valueaddvc.com/blog/ai-coding-productivity-2026-what-metr-github-and-dora-studies-actually-show) — valueaddvc.com, blog, n.d., quality 0.58 _(retrieved, not cited)_
- **[S2]** [Gen AI Impact: 56% Faster Dev Tasks](https://cut-the-saas.com/ai/generative-ai-impact-on-developers-increased-productivity-and-faster-task-completion) — cut-the-saas.com, other, n.d., quality 0.62 _(retrieved, not cited)_
- **[S3]** [Randomized controlled study finds GitHub Copilot enables ...](https://ai-best-practices.com/case-studies/randomized-controlled-study-finds-github-copilot-enables-developers-to-c) — ai-best-practices.com, other, n.d., quality 0.56 _(retrieved, not cited)_
- **[S4]** [The Impact of AI Coding in 2026: Developer Productivity Revolution with 90% AI-Generated Code](https://trigidigital.com/blog/ai-coding-impact-2026) — trigidigital.com, blog, n.d., quality 0.56 _(retrieved, not cited)_
- **[S5]** [Medium](https://ingoeichhorst.medium.com/state-of-ai-coding-efficiency-2026-1abfa0ab7434) — ingoeichhorst.medium.com, blog, n.d., quality 0.56
- **[S6]** [Developer Productivity With and Without GitHub Copilot](https://arxiv.org/html/2509.20353v2) — arxiv.org, academic, n.d., quality 0.96 _(retrieved, not cited)_
- **[S7]** [Evidence from a Field Experiment with GitHub Copilot](https://mit-genai.pubpub.org/pub/v5iixksv) — mit-genai.pubpub.org, other, n.d., quality 0.60 _(retrieved, not cited)_
- **[S8]** [Does AI-Assisted Coding Deliver? A Difference-in-Differences Study of Cursor’s Impact on Software Projects](https://arxiv.org/html/2511.04427v2) — arxiv.org, academic, n.d., quality 0.95
- **[S10]** [The Impact of AI on Developer Productivity](https://www.alphaxiv.org/abs/2302.06590) — alphaxiv.org, other, n.d., quality 0.59

## Citation verification

- Evidence references: 5, 5 resolved to citable evidence (100%). Citation markers are derived from those references by the engine, so citation integrity is a structural invariant rather than a measurement.
- Evidence-owing claims carrying a citation: 100% of 5
- Entailment checked against each claim's own evidence, over every eligible claim: 5 supported, 4 partially supported, 31 unsupported, 0 not checked
- Retrieved but never cited: S1, S2, S3, S4, S6, S7

<details><summary>Open citation issues</summary>

- `unsupported_claim` every cited quote failed a deterministic guard: atomicity — Empirical studies measure productivity using self-assessments, measured output, accepted pull requests, and codebase-lev
- `unsupported_claim` every cited quote failed a deterministic guard: atomicity — In one controlled experiment, developers using Copilot completed a standardized HTTP-server task roughly 56% faster than
- `unsupported_claim` every cited quote failed a deterministic guard: atomicity — In METR’s randomized trial, developers allowed to use AI took 19% longer to complete tasks than the control group.
- `unsupported_claim` every cited quote failed a deterministic guard: framing — A randomized within-subject Copilot experiment found no statistically significant difference in task-completion time.
- `unsupported_claim` best entailment 0.000 from S1-e2 is below the 0.98 support threshold — The evidence includes controlled experiments reporting faster completion, slower completion, and no statistically signif
- `unsupported_claim` every cited quote failed a deterministic guard: atomicity, attribution — A field experiment’s estimated changes in pull requests per week were described as suggestive, and its authors reported 
- `unsupported_claim` every cited quote failed a deterministic guard: atomicity, framing — Some reported speed and output gains are accompanied by evidence of quality-related costs, but the supplied findings do 
- `unsupported_claim` every cited quote failed a deterministic guard: atomicity — One experiment measured task success as the percentage of participants who completed the task and measured task-completi
- `unsupported_claim` every cited quote failed a deterministic guard: atomicity — A maintainability study assessed modification time, CodeScene Code Health, test coverage, and perceived productivity.
- `unsupported_claim` every cited quote failed a deterministic guard: atomicity — A field experiment measured pull requests, commits, and builds as proxies for completed tasks.
- `unsupported_claim` every cited quote failed a deterministic guard: atomicity — A field experiment was designed to measure output quantity, review time, and several measures of work quality.
- `unsupported_claim` every cited quote failed a deterministic guard: atomicity — In that transcript-based study, self-reported daily estimates consistently undershot the measured gains.
- `unsupported_claim` every cited quote failed a deterministic guard: atomicity — In the Copilot HTTP-server experiment, the treatment group completed the task in an average of 71 minutes, compared with
- `unsupported_claim` best entailment 0.001 from S3-e3 is below the 0.98 support threshold — The Copilot HTTP-server experiment reported a 55.8% speed improvement, with p = 0.0017 and a 95% confidence interval of 
- `unsupported_claim` every cited quote failed a deterministic guard: atomicity — In a separate randomized trial, developers allowed to use AI took 19% longer to finish tasks than the control group.
- `unsupported_claim` every cited quote failed a deterministic guard: framing — A randomized within-subject experiment found no statistically significant difference in task-completion time with Copilo
- `unsupported_claim` every cited quote failed a deterministic guard: atomicity — In the Copilot HTTP-server experiment, task completion rates were 7 percentage points higher in the treatment group, but
- `unsupported_claim` every cited quote failed a deterministic guard: attribution — A field experiment reported suggestive evidence of 12.92%–21.83% more pull requests per week among participating Microso
- `unsupported_claim` every cited quote failed a deterministic guard: attribution — The same field experiment reported suggestive evidence of 7.51%–8.69% more pull requests per week among participating Ac
- `unsupported_claim` every cited quote failed a deterministic guard: framing — The field-experiment authors said their estimates were not very precise because of low Copilot compliance in the Microso

</details>

## Run metrics

| Metric | Value |
| --- | --- |
| Mode | cloud |
| Research rounds | 2 |
| Stopped because | stopped after round 2 |
| Search queries | 12 |
| Unique sources | 10 |
| Usable sources | 9 |
| Distinct domains | 8 |
| Fetches avoided by dedup | 14 |
| Evidence items | 54 |
| Quotes verbatim (exact-normalised) | 98% |
| Quotes fuzzy (excluded from citation) | 0% |
| LLM calls | 18 |
| Tokens (in/out) | 40,222 / 25,019 |
| Estimated cost | $0.0165 |
| Duration | 237.1s |

---

_Generated by Agentic Research Engine on 2026-09-27 22:48 UTC._
