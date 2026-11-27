# Model Performance Comparison and Top Two Selection

## 1. Objective

This document addresses Issue #39 by comparing the fraud detection
models recorded in the shared MLflow experiment and recommending
the top two candidates for model registration.

The stakeholder prioritises high precision while maintaining
reasonable recall to reduce unnecessary fraud investigations
without overlooking too many fraudulent orders.

No numerical precision or recall requirement was specified
by the stakeholder.

## 2. Experiment Inventory

All 16 official model runs were retrieved from the shared
SP-buy-Fraud-Detection MLflow experiment (Experiment ID: 2).

| Developer | Model Configurations | Safe | Full |
|---|---|---:|---:|
| Abby | Logistic Regression, Random Forest | 2 | 2 |
| Yueliang | Logistic Regression, HistGradientBoosting | 2 | 2 |
| Emi | Decision Tree, MLP | 2 | 2 |
| Chu | Extra Trees, LightGBM | 2 | 2 |
| Total | 8 configurations | 8 | 8 |

All 16 runs were finished and contained the required evaluation
metrics, model identifiers and dataset metadata.

The dataset SHA-256 checksums were consistent across runs using
the same dataset version.

## 3. Evaluation Dataset

A consistent chronological split was used to reflect future
fraud assessment more realistically than a random split.

| Split | Orders | Fraud Orders |
|---|---:|---:|
| Training | 665,876 | 67,594 |
| Validation | 139,874 | 18,993 |
| Test | 144,122 | 17,835 |
| Total | 949,872 | 104,422 |

The validation set was used for threshold selection and
candidate ranking. The selected thresholds were locked before
the new test evaluation.

Original test results had already been examined during the
earlier model experiments. Therefore, the test set should not
be described as a completely unseen blind evaluation.

## 4. Safe vs Full Feature Comparison

The Safe dataset contains 31 features, while the Full dataset
contains 41 features, including 10 customer snapshot features
whose availability at assessment time remains uncertain.

The refund_value feature is excluded in accordance with
stakeholder requirements.

| Developer | Algorithm | Safe Test AP | Full Test AP | Improvement |
|---|---|---:|---:|---:|
| Abby | Logistic Regression | 0.3981 | 0.4501 | +0.0520 |
| Abby | Random Forest | 0.4177 | 0.5076 | +0.0899 |
| Chu | Extra Trees | 0.4350 | 0.4912 | +0.0562 |
| Chu | LightGBM | 0.4247 | 0.4971 | +0.0724 |
| Emi | Decision Tree | 0.4173 | 0.4536 | +0.0363 |
| Emi | MLP | 0.3995 | 0.4257 | +0.0263 |
| Yueliang | HistGradientBoosting | 0.4251 | 0.5089 | +0.0837 |
| Yueliang | Logistic Regression | 0.3975 | 0.4493 | +0.0517 |

All eight Full models achieved higher test average precision
than their corresponding Safe models.

Random Forest recorded the largest improvement, followed by
HistGradientBoosting.

These findings suggest that the additional features contain
predictive information. However, improved performance alone
does not establish whether data leakage occurred.

The Full models remain exploratory and are excluded from
model selection until feature availability can be verified.

Safe models are selection candidates, but are not certified
as production-ready.

Figure: outputs/model_comparison/plots/01_safe_vs_full_ap.png

## 5. Consistent Threshold Selection Methodology

The original experiments used different decision threshold rules:

| Developer | Original Threshold Rule |
|---|---|
| Abby | Fixed threshold of 0.5 |
| Yueliang | Maximum validation recall at minimum 70% precision |
| Emi | Maximum validation F1 |
| Chu | Maximum validation F1 |

Directly ranking the original precision or recall values would
not provide a fair comparison because the models were evaluated
at different operating points.

All eight Safe models were therefore reloaded from MLflow
and evaluated on the same validation dataset.

Three precision targets were investigated:

- 60%: relaxed precision scenario.
- 70%: primary comparison policy.
- 80%: strict precision scenario.

The 70% precision target is a proposed technical comparison
policy, not a stakeholder-mandated business requirement.

For each model, the selected decision threshold maximises
validation recall while maintaining the required minimum
validation precision.

Candidates are ranked using:

1. Highest validation recall among eligible candidates.
2. Higher validation precision as the first tie-breaker.
3. Higher validation average precision as the second tie-breaker.
4. Run name as a deterministic final tie-breaker.

The original MLflow metrics and decision thresholds remain
unchanged.

## 6. Primary Model Comparison

All eight Safe models achieved at least 70% validation precision.

| Rank | Developer | Algorithm | Precision | Recall | F1 | AP |
|---:|---|---|---:|---:|---:|---:|
| 1 | Chu | Extra Trees | 70.04% | 16.64% | 0.2690 | 0.5136 |
| 2 | Emi | Decision Tree | 70.23% | 15.27% | 0.2509 | 0.5160 |
| 3 | Yueliang | HistGradientBoosting | 70.08% | 14.11% | 0.2349 | 0.5340 |
| 4 | Chu | LightGBM | 70.02% | 13.29% | 0.2235 | 0.5338 |
| 5 | Abby | Logistic Regression | 70.02% | 12.62% | 0.2138 | 0.4586 |
| 6 | Yueliang | Logistic Regression | 70.01% | 11.90% | 0.2034 | 0.4610 |
| 7 | Abby | Random Forest | 70.03% | 9.97% | 0.1745 | 0.5289 |
| 8 | Emi | MLP | 70.45% | 1.90% | 0.0369 | 0.5099 |

Extra Trees achieved the highest recall under the primary
precision constraint, followed by Decision Tree.

Although HistGradientBoosting achieved the highest overall
validation average precision, its recall at the selected
70% precision operating point was lower.

Average precision measures ranking performance across
thresholds, rather than performance at a single decision
threshold.

The selection therefore prioritises performance at the
defined business-oriented operating point.

Figure: outputs/model_comparison/plots/02_primary_model_ranking.png

## 7. Precision Target Sensitivity Analysis

The table reports maximum validation recall at each minimum
precision target.

| Developer | Algorithm | 60% Precision | 70% Precision | 80% Precision |
|---|---|---:|---:|---:|
| Abby | Logistic Regression | 26.42% | 12.62% | 0.57% |
| Abby | Random Forest | 41.16% | 9.97% | 4.39% |
| Chu | Extra Trees | 32.64% | 16.64% | 1.84% |
| Chu | LightGBM | 41.90% | 13.29% | 6.52% |
| Emi | Decision Tree | 41.14% | 15.27% | 3.26% |
| Emi | MLP | 38.95% | 1.90% | 0.82% |
| Yueliang | HistGradientBoosting | 41.77% | 14.11% | 6.36% |
| Yueliang | Logistic Regression | 26.65% | 11.90% | 2.92% |

The results demonstrate a substantial precision-recall trade-off.

At 60% precision, LightGBM and HistGradientBoosting achieved
the highest recall.

At 70% precision, Extra Trees and Decision Tree ranked first
and second.

At 80% precision, all models achieved less than 7% recall.

No single model dominates across all three operating points.

The alternative scenarios provide evidence for future
stakeholder discussions. They do not replace the previously
defined primary selection policy.

Figure: outputs/model_comparison/plots/03_threshold_sensitivity.png

## 8. Selected Models

### 8.1 Proposed Champion: Extra Trees (Safe)

- Developer: Chu
- Run name: chu_extra_trees_safe_v1
- Run ID: 2ad5cf74f0f040d9841767e94331af10
- Logged model URI: models:/m-a28a6f67e51042948058e9d96bb2c509
- Original threshold: 0.6273964956169535
- Selected threshold: 0.78125424937575
- Validation precision: 70.04%
- Validation recall: 16.64%
- Validation F1: 0.2690
- Validation AP: 0.5136
- Production eligible: false

Extra Trees is recommended because it identifies the largest
number of fraudulent validation orders while meeting the
primary precision constraint.

It correctly identifies 3,161 fraudulent validation orders
and generates 1,352 false positives.

### 8.2 Proposed Challenger: Decision Tree (Safe)

- Developer: Emi
- Run name: emi_decision_tree_safe_v1
- Run ID: f6d5330039164615bdd37dea09df56d0
- Logged model URI: models:/m-8cf5204701264128b8e9eb066157ea78
- Original threshold: 0.14
- Selected threshold: 0.5660377358490566
- Validation precision: 70.23%
- Validation recall: 15.27%
- Validation F1: 0.2509
- Validation AP: 0.5160
- Production eligible: false

Decision Tree is recommended as the second candidate because
it achieves the second-highest recall under the primary
precision constraint.

It correctly identifies 2,901 fraudulent validation orders
and generates 1,230 false positives.

Its single-tree structure also offers a simpler basis for
model interpretation than an ensemble of trees.

## 9. Generalisation and Test Evaluation

The two selected models and their full-precision decision
thresholds were saved before the new test evaluation.

| Metric | Extra Trees | Decision Tree |
|---|---:|---:|
| Validation precision | 70.04% | 70.23% |
| Test precision | 60.62% | 58.47% |
| Validation recall | 16.64% | 15.27% |
| Test recall | 21.33% | 18.75% |
| Test F1 | 0.3156 | 0.2839 |
| Validation AP | 0.5136 | 0.5160 |
| Test AP | 0.4350 | 0.4173 |
| Precision change | -9.42 pp | -11.75 pp |
| Recall change | +4.69 pp | +3.48 pp |
| AP change | -0.0786 | -0.0986 |
| Test precision target met | No | No |

Both models experienced a reduction in precision and average
precision on the later test period.

Extra Trees showed a smaller precision reduction and higher
test recall than Decision Tree.

The decline suggests weaker generalisation to the later
evaluation period. Changes in fraud prevalence or feature
distributions are possible explanations, but additional
analysis is required before concluding that data drift occurred.

The model ranking and selected thresholds were not changed
after reviewing these test results.

Figure: outputs/model_comparison/plots/04_top_two_generalisation.png

## 10. Operational Considerations

| Test Metric | Extra Trees | Decision Tree |
|---|---:|---:|
| Total orders | 144,122 | 144,122 |
| Fraud orders detected | 3,805 | 3,344 |
| False positives | 2,472 | 2,375 |
| Fraud orders missed | 14,030 | 14,491 |
| Total flagged orders | 6,277 | 5,719 |
| Observed batch inference time | 0.1860s | 0.0189s |

Extra Trees detects 461 more fraudulent orders while generating
97 additional false positives compared with Decision Tree.

Decision Tree is simpler and recorded a lower inference time
during this evaluation.

The inference timings are single-run observations and should
not be treated as formal performance benchmarks or evidence
that the complete ETL pipeline meets its runtime requirement.

Both candidates miss a substantial proportion of fraudulent
orders. Consequently, neither should be presented as an
adequately validated production fraud detection system.

## 11. Verification Results

| Verification | Result |
|---|---|
| Official MLflow runs retrieved | 16/16 |
| Safe-Full model pairs | 8/8 |
| Required original metrics retrieved | 192/192 |
| Safe dataset checksum verification | PASS |
| Safe models reloaded | 8/8 |
| Original validation metrics reproduced | 48/48 |
| Precision target evaluations | 24/24 |
| Selected models evaluated on test | 2/2 |
| Original selected-model test metrics reproduced | 12/12 |
| Comparison charts generated | 4/4 |
| Original MLflow records modified | 0 |
| Additional models trained | 0 |

## 12. Reproduction Instructions

Configure the shared MLflow tracking URI and individual
authentication credentials through environment variables.

Run the following commands from the repository root:

    python src/model_comparison.py
    python src/model_threshold_evaluation.py
    python src/model_selection.py
    python src/model_comparison_plots.py

The default output directory is outputs/model_comparison.

Generated outputs:

- all_model_metrics.csv
- safe_full_comparison.csv
- safe_model_comparison.csv
- threshold_sensitivity.csv
- top_two_models.json
- top_two_test_results.csv
- plots/01_safe_vs_full_ap.png
- plots/02_primary_model_ranking.png
- plots/03_threshold_sensitivity.png
- plots/04_top_two_generalisation.png

These scripts target the 16 official CA1 baseline runs.
They are not a general-purpose selection mechanism for
arbitrary future retraining batches.

## 13. Limitations and Recommendations

- The 70% precision target is a proposed comparison policy
  and requires stakeholder confirmation before operational use.
- Both selected models fail to maintain 70% test precision.
- Recall remains low under the primary precision constraint.
- Safe feature availability and production readiness require
  further validation.
- Full models are excluded because the assessment-time
  availability of additional snapshot features is uncertain.
- Ranking is sensitive to the selected precision target.
- Original test results were already inspected during earlier
  experiments, limiting claims of independent blind evaluation.
- Inference time was measured once and is not a formal benchmark.
- Future retraining should repeat candidate comparison using
  a consistent policy and an appropriate chronological split.
- Future monitoring should examine precision, recall, feature
  distributions and investigation workload.
- The acceptable precision-recall trade-off should be
  confirmed with the stakeholder.

## 14. Final Recommendation

Register the following Safe models for the CA1 model
management demonstration under Issue #40:

1. Extra Trees as the proposed Champion.
2. Decision Tree as the proposed Challenger.

Their exact selected decision thresholds must be preserved
alongside their registered model versions and used by the
subsequent fraud risk scoring workflow.

Both models must retain production_eligible=false.

Registration and champion promotion are demonstration-level
model management decisions and do not constitute approval
for production deployment.

## 15. Automated Testing Results

| Test Category | Result |
|---|---|
| Model comparison unit tests | 28 passed |
| Combined regression tests | 47 passed, 1 skipped |
| Regression subtests | 22 passed |
| Failed tests | 0 |
| Combined regression runtime | 16.98 seconds |
| Selection manifest SHA-256 verification | PASS |

The automated tests cover official run inventory validation,
precision-constrained threshold selection, candidate eligibility,
deterministic ranking, independence from test metrics and
protection against accidental selection manifest overwrites.

The database integration test was skipped because a separate
ETL_TEST_DATABASE was not configured.

The 14 warnings were dependency deprecation warnings from
Matplotlib and Pyparsing.

These tests cover the main comparison and selection rules,
but do not represent exhaustive production-level test coverage.

The locked baseline selection manifest is preserved at
docs/top_two_models.json. Its SHA-256 checksum matches the
original generated selection manifest.
