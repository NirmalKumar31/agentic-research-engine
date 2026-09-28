# Empirical measures of AI coding assistant productivity

**Question:** What is the measured evidence on whether AI coding assistants improve developer productivity?

## Source excerpts

_No synthesized claim passed evidence verification. Showing exact source excerpts instead. These are verbatim quotations, not findings, and no conclusion has been drawn from them._

- "The treated group completed the task 55.8% faster (95% confidence interval: 21-89%)." — **[S1]** The Impact of AI on Developer Productivity: Evidence from GitHub Copilot
- "Conditioning on completing the task, the average completion time from the treated group is 71.17 minutes and 160.89 minutes for the control group." — **[S1]** The Impact of AI on Developer Productivity: Evidence from GitHub Copilot
- "Participants were instructed to write an HTTP server in JavaScript—the treatment group would use GitHub Copilot to complete the task, while the control group would not." — **[S1]** The Impact of AI on Developer Productivity: Evidence from GitHub Copilot
- "The results show that less experienced developers (years of professional coding), developers with heavy coding load (hours of coding per day), and older developers (developers aged between 25 and 44) benefit more from Copilot." — **[S1]** The Impact of AI on Developer Productivity: Evidence from GitHub Copilot
- "We also find that the treated group’s success rate is 7 percentage points higher than the control group, but the estimate is not statistically significant, with a 95% confidence interval of [-0.11, 0.25]." — **[S1]** The Impact of AI on Developer Productivity: Evidence from GitHub Copilot
- "We calculated two metrics as a measure of performance for each group: task success and task completion time." — **[S1]** The Impact of AI on Developer Productivity: Evidence from GitHub Copilot
- "Developers with access to Copilot completed the programming task 55.8% faster than those without access (95% confidence interval: 21-89%)." — **[S3]** The Impact of AI on Developer Productivity: Evidence from GitHub Copilot | alphaXiv
- "The experiment involved a single programming task in JavaScript. Future research should explore different programming languages and more complex tasks." — **[S3]** The Impact of AI on Developer Productivity: Evidence from GitHub Copilot | alphaXiv
- "The experiment involved a single programming task in JavaScript. Future research should explore different programming languages and more complex tasks." — **[S3]** The Impact of AI on Developer Productivity: Evidence from GitHub Copilot | alphaXiv
- "Developers with less programming experience benefited more from Copilot than those with extensive experience." — **[S3]** The Impact of AI on Developer Productivity: Evidence from GitHub Copilot | alphaXiv
- "While the treatment group had a higher success rate than the control group, this difference was not statistically significant." — **[S3]** The Impact of AI on Developer Productivity: Evidence from GitHub Copilot | alphaXiv
- "The study focused on task completion time rather than code quality. Additional research is needed to assess how AI tools affect code quality, maintainability, and security." — **[S3]** The Impact of AI on Developer Productivity: Evidence from GitHub Copilot | alphaXiv

## Limitations

- The supplied evidence does not establish whether measured productivity effects persist beyond short tasks or initial use.
- The supplied evidence does not establish the general effects of AI coding assistants across task types: one cited experiment used a single JavaScript task, and the available task comparisons are limited.
- The supplied evidence does not establish whether time savings generally come with changes in code correctness, maintainability, security, or rework.
- Only limited evidence was found for: How strong and generalizable is the evidence, considering study design, sample size, controlled versus real-world settings, and whether measured gains persist beyond short tasks or initial use.
- Only limited evidence was found for: How do measured effects differ across task types and difficulty, such as code generation, debugging, documentation, or unfamiliar codebases.
- Only limited evidence was found for: How do findings differ across productivity and quality measures—such as time, task completion, code correctness, defects, maintainability, and rework—and do gains on one measure come with losses on another.
- 6 generated claim(s) were excluded because the cited evidence did not support them, or because verification did not reach them within this run's budget.
- 1 reported disagreement(s) were excluded because the evidence did not support both sides as stated.

## Sources

- **[S1]** [The Impact of AI on Developer Productivity: Evidence from GitHub Copilot](https://arxiv.org/html/2302.06590v1) — arxiv.org, academic, n.d., quality 0.97 _(retrieved, not cited)_
- **[S2]** [Medium](https://ingoeichhorst.medium.com/state-of-ai-coding-efficiency-2026-1abfa0ab7434) — ingoeichhorst.medium.com, blog, n.d., quality 0.56 _(retrieved, not cited)_
- **[S3]** [The Impact of AI on Developer Productivity: Evidence from GitHub Copilot | alphaXiv](https://www.alphaxiv.org/abs/2302.06590) — alphaxiv.org, other, n.d., quality 0.61 _(retrieved, not cited)_
- **[S4]** [Research: quantifying GitHub Copilot’s impact on developer productivity and happiness - The GitHub Blog](https://github.blog/news-insights/research/research-quantifying-github-copilots-impact-on-developer-productivity-and-happiness) — github.blog, other, n.d., quality 0.61 _(retrieved, not cited)_
- **[S5]** [The Productivity Effects of Generative AI: Evidence from a Field Experiment with GitHub Copilot · From Novel Chemicals to Opera](https://mit-genai.pubpub.org/pub/v5iixksv) — mit-genai.pubpub.org, other, n.d., quality 0.61 _(retrieved, not cited)_
- **[S6]** [The Impact of AI Coding in 2026: Developer Productivity ...](https://trigidigital.com/blog/ai-coding-impact-2026) — trigidigital.com, blog, n.d., quality 0.55 _(retrieved, not cited)_

## Citation verification

- Evidence references: 0, 0 resolved to citable evidence (n/a, no references). Citation markers are derived from those references by the engine, so citation integrity is a structural invariant rather than a measurement.
- Evidence-owing claims carrying a citation: n/a, no substantive claims were published
- Entailment checked against each claim's own evidence, over every eligible claim: 0 supported, 2 partially supported, 4 unsupported, 0 not checked
- Retrieved but never cited: S1, S2, S3, S4, S5, S6

<details><summary>Open citation issues</summary>

- `partially_supported_claim` best entailment 0.976 from S2-e1 is below the 0.98 support threshold — Reported productivity effects vary across studies, ranging from a 20% slowdown to a 100% speedup.
- `unsupported_claim` every cited quote failed a deterministic guard: attribution — In two field experiments, estimates of changes in pull requests completed per week ranged from 12.92% to 21.83% at Micro
- `unsupported_claim` every cited quote failed a deterministic guard: atomicity, numeric — In one controlled experiment, developers with Copilot completed the task 55.8% faster than developers without it, with a
- `unsupported_claim` best entailment 0.007 from S2-e3 is below the 0.98 support threshold — Reported gains were lower for brownfield than greenfield work at both low and high complexity.
- `partially_supported_claim` best entailment 0.948 from S3-e4 is below the 0.98 support threshold — One source reports that developers with less programming experience benefited more from Copilot than developers with ext
- `unsupported_claim` every cited quote failed a deterministic guard: atomicity — In one study, the estimated task-success rate was 7 percentage points higher with Copilot, but the estimate was not stat
- `unsupported_claim` contradiction 0 left side: every cited quote failed a deterministic guard: framing — A cited controlled study found no statistically significant difference in completion time.
- `unsupported_claim` contradiction 0 right side: best entailment 0.001 from S3-e1 is below the 0.98 support threshold — A different controlled experiment found that developers with Copilot completed the task 55.8% faster.

</details>

## Run metrics

| Metric | Value |
| --- | --- |
| Mode | cloud |
| Research rounds | 1 |
| Stopped because | stopped after round 1 |
| Search queries | 6 |
| Unique sources | 6 |
| Usable sources | 6 |
| Distinct domains | 6 |
| Fetches avoided by dedup | 7 |
| Evidence items | 34 |
| Quotes verbatim (exact-normalised) | 100% |
| Quotes fuzzy (excluded from citation) | 0% |
| LLM calls | 12 |
| Tokens (in/out) | 27,266 / 13,491 |
| Estimated cost | $0.0095 (+0 calls of unknown price) |
| Duration | 144.4s |

---

_Generated by Agentic Research Engine on 2026-09-28 04:53 UTC._
