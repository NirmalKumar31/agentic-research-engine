# Modern Approaches for Detecting Fraud in Highly Imbalanced Transaction Datasets: A Comparative Analysis

## Summary

FraudX AI combines random forest and XGBoost as baseline models for fraud detection in highly imbalanced datasets. [S1] Decision Tree and Logistic regression have good results with imbalanced data, while SVM benefits from balancing processes to improve precision by up to 54%. [S5] Stratified cross validation is the foundation strategy for handling imbalanced datasets in fraud detection. [S2] *(synthesis)* The number of fraudulent transactions is typically much lower than legitimate transactions in fraud detection scenarios. [S3] AUC-PR and recall metrics are prioritized over traditional accuracy-based metrics for evaluating fraud detection models in imbalanced datasets. [S1] In scenarios with high data imbalance, metrics such as Precision, Recall, F-score, and AUC are typically used for evaluation. [S5]

## Key findings

- FraudX AI combines random forest and XGBoost as baseline models for fraud detection in highly imbalanced datasets. [S1]
- Decision Tree and Logistic regression have good results with imbalanced data, while SVM benefits from balancing processes to improve precision by up to 54%. [S5]
- Stratified cross validation is the foundation strategy for handling imbalanced datasets in fraud detection. [S2] *(synthesis)*
- The number of fraudulent transactions is typically much lower than legitimate transactions in fraud detection scenarios. [S3]
- AUC-PR and recall metrics are prioritized over traditional accuracy-based metrics for evaluating fraud detection models in imbalanced datasets. [S1]
- In scenarios with high data imbalance, metrics such as Precision, Recall, F-score, and AUC are typically used for evaluation. [S5]

## Model Selection for Fraud Detection in Highly Imbalanced Datasets

FraudX AI combines random forest and XGBoost as baseline models for fraud detection in highly imbalanced datasets. [S1]

Decision Tree and Logistic regression have good results with imbalanced data, while SVM benefits from balancing processes to improve precision by up to 54%. [S5]

## Model Evaluation in Imbalanced Datasets

Stratified cross validation is the foundation strategy for handling imbalanced datasets in fraud detection. [S2] *(synthesis)*

AUC-PR and recall metrics are prioritized over traditional accuracy-based metrics for evaluating fraud detection models in imbalanced datasets. [S1]

In scenarios with high data imbalance, metrics such as Precision, Recall, F-score, and AUC are typically used for evaluation. [S5]

## Limitations

- The evidence does not provide specific information on hidden costs of implementing fraud detection models in real-time transaction systems.
- The evidence does not detail failure modes when fraud detection models encounter new types of fraud attacks.
- The evidence does not address how regulatory requirements impact the design and evaluation of fraud detection models in different regions.
- The evidence does not specify trade-offs between model accuracy and real-time processing requirements in fraud detection systems.
- 7 extracted finding(s) were excluded because their quotes could not be located in the source text

## Sources

- **[S1]** [FraudX AI: An Interpretable Machine Learning Framework for Credit Card Fraud Detection on Imbalanced Datasets](https://www.mdpi.com/2073-431X/14/4/120) — mdpi.com, other, n.d., quality 0.63
- **[S2]** [Cross Validation Strategies for Imbalanced Datasets - ML Journey](https://mljourney.com/cross-validation-strategies-for-imbalanced-datasets) — mljourney.com, other, n.d., quality 0.63
- **[S3]** [Evaluation Metrics for Imbalanced Datasets](https://schneppat.com/evaluation-metrics.html) — schneppat.com, other, n.d., quality 0.62
- **[S4]** [Credit Card Fraud Detection - Dealing with Imbalanced Datasets in Machine Learning](https://www.youtube.com/watch?v=M_Cu7r9gik4) — youtube.com, other, n.d., quality 0.62 _(retrieved, not cited)_
- **[S5]** [Imbalanced classification in Fraud Detection](https://medium.com/data-reply-it-datatech/imbalanced-classification-in-fraud-detection-8f63474ff8c7) — medium.com, blog, n.d., quality 0.56

## Citation verification

- Evidence references: 17, 17 resolved to citable evidence (100%). Citation markers are derived from those references by the engine, so citation integrity is a structural invariant rather than a measurement.
- Evidence-owing claims carrying a citation: 100% of 17
- Entailment checked against each claim's own evidence, over a sample of 10 of 17 eligible claims: 10 supported, 0 partially supported, 0 unsupported, 7 not checked
- Retrieved but never cited: S4

## Run metrics

| Metric | Value |
| --- | --- |
| Mode | local |
| Research rounds | 1 |
| Stopped because | coverage judged sufficient |
| Search queries | 5 |
| Unique sources | 5 |
| Usable sources | 5 |
| Distinct domains | 5 |
| Fetches avoided by dedup | 3 |
| Evidence items | 27 |
| Quotes verbatim (exact-normalised) | 74% |
| Quotes fuzzy (excluded from citation) | 7% |
| LLM calls | 20 |
| Tokens (in/out) | 19,508 / 6,318 |
| Estimated cost | $0.0000 |
| Duration | 1095.7s |

---

_Generated by Agentic Research Engine on 2026-09-23 01:38 UTC._
