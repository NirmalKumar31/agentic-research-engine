# Main factors that contribute to machine-learning overfitting

**Question:** What are the main causes of overfitting in machine learning?

## Summary

Overfitting often arises when a model is excessively complex relative to the size and diversity of its training data. [S3] Overfitting can occur when the training data is imbalanced. [S3] Without regularization, models are free to fit training data too closely, increasing the risk of overfitting. [S3]

## Additional supported drivers

Training a model for too many epochs can lead to overfitting. [S3]

Too many features can potentially magnify noise and result in overfitting. [S4]

## Limitations

- The evidence does not establish data leakage as a cause of overfitting or explain how it might produce misleadingly strong validation results.
- The evidence on feature dimensionality is limited: it supports a potential overfitting risk from too many features, but does not establish that high dimensionality causes misleading validation results.
- The evidence does not establish repeated tuning against the same validation data as a cause of overfitting.
- The evidence does not compare the relative importance of the supported drivers.
- Only limited evidence was found for: How do data leakage and high-dimensional feature spaces create misleadingly strong training or validation results and poor performance on truly unseen data.
- Only limited evidence was found for: How do weak or inappropriate regularization and model-selection practices contribute to overfitting.
- Only limited evidence was found for: How can prolonged training and repeated tuning against the same validation data lead to overfitting.
- 1 generated claim(s) were excluded because the cited evidence did not support them, or because verification did not reach them within this run's budget.
- The evidence did not establish how a driver produces the effect.
- The evidence did not establish whether the evidence establishes causation or only association.
- More than one published claim fills the candidate_drivers slot; they may repeat each other.

## Sources

- **[S1]** [Overfitting  |  Machine Learning  |  Google for Developers](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting) — developers.google.com, official_docs, n.d., quality 0.98 _(retrieved, not cited)_
- **[S2]** [Underfitting and Overfitting in ML - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/underfitting-and-overfitting-in-machine-learning) — geeksforgeeks.org, other, n.d., quality 0.63 _(retrieved, not cited)_
- **[S3]** [Understanding Overfitting in Machine Learning: Causes, Impacts, and Solutions | Lenovo US](https://www.lenovo.com/us/en/knowledgebase/understanding-overfitting-in-machine-learning-causes-impacts-and-solutions) — lenovo.com, other, n.d., quality 0.64
- **[S4]** [Model Complexity & Overfitting in Machine Learning - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/model-complexity-overfitting-in-machine-learning) — geeksforgeeks.org, other, n.d., quality 0.63
- **[S5]** [Overfitting: Model complexity  |  Machine Learning  |  Google for Developers](https://developers.google.com/machine-learning/crash-course/overfitting/model-complexity) — developers.google.com, official_docs, n.d., quality 0.97 _(retrieved, not cited)_
- **[S6]** [Overfitting: Causes and Remedies | Towards AI](https://towardsai.com/p/l/overfitting-causes-and-remedies) — towardsai.com, other, n.d., quality 0.63 _(retrieved, not cited)_

## Citation verification

- Evidence references: 5, 5 resolved to citable evidence (100%). Citation markers are derived from those references by the engine, so citation integrity is a structural invariant rather than a measurement.
- Evidence-owing claims carrying a citation: 100% of 5
- Entailment checked against each claim's own evidence, over every eligible claim: 5 supported, 0 partially supported, 1 unsupported, 0 not checked
- Retrieved but never cited: S1, S2, S5, S6

<details><summary>Open citation issues</summary>

- `unsupported_claim` every cited quote failed a deterministic guard: atomicity — When a training dataset is too small, a model may memorize specific details rather than identify generalizable patterns,
- `unsupported_claim` every cited quote failed a deterministic guard: atomicity — Overfitting can occur when the training data is imbalanced, with some classes overrepresented and others underrepresente

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
| Distinct domains | 4 |
| Fetches avoided by dedup | 6 |
| Evidence items | 25 |
| Quotes verbatim (exact-normalised) | 100% |
| Quotes fuzzy (excluded from citation) | 0% |
| LLM calls | 14 |
| Tokens (in/out) | 25,662 / 12,485 |
| Estimated cost | $0.0088 (8 responses used a token category with no recorded rate) |
| Duration | 147.5s |

---

_Generated by Agentic Research Engine on 2026-09-30 22:13 UTC._
