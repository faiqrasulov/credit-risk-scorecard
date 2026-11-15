# Model card: German Credit WoE scorecard

All numbers below come from `make run` and are stored in
[`reports/metrics.json`](reports/metrics.json).

## Model details

- **What it is:** a logistic regression on Weight-of-Evidence (WoE) binned features, scaled to
  points: 600 points at good:bad odds of 50:1, 20 points to double the odds (PDO). The score is
  the sum of one points value per feature, so every decision can be traced to a table
  ([`reports/scorecard.md`](reports/scorecard.md)).
- **Size:** 11 features, 46 bins (final model, fitted on all 1,000 applicants). Candidate
  features with Information Value below 0.02 are dropped inside each training fold, and any
  feature whose coefficient has the wrong sign would be dropped too (none did in the final fit).
- **Decision rule:** reject when PD >= 0.1495, equivalently score <= 537.3. The cutoff minimises
  the expected cost of the German Credit cost matrix (approving a bad applicant costs 5,
  rejecting a good one costs 1) on the pooled out-of-fold predictions of 5 x 5 repeated
  stratified cross-validation.
- **Outputs (API `POST /score`):** probability of default, score points, approve/reject, and the
  three features that cost the applicant the most points relative to that feature's best bin
  (adverse-action reason codes).
- **Version / owner:** 0.1.0, Faiq Rasulov (portfolio project), 2026. MIT licence.

## Intended use

A worked, reproducible example of how a bank-style application scorecard is built, validated,
explained and audited, and how it compares with a gradient-boosting model. It is intended for
learning and for discussing modelling choices.

**Out of scope:** any real lending decision. The model is trained on 1,000 credits from one
German bank in 1973-1975, and its PDs are on the scale of an oversampled data set (see Data).

## Data

- Statlog German Credit (Hofmann, 1994; UCI Machine Learning Repository), loaded from OpenML
  (`data_id=31`): 1,000 credits, 20 attributes, 700 good and 300 bad.
- **Oversampled bads.** Grömping (2019) reports that the sample is stratified with bad credits
  heavily oversampled; the bank's actual bad rate was around 5%, not 30%. PDs, approval rates and
  the cost-optimal cutoff therefore apply to the 30%-bad sample only. Using the model on a real
  applicant flow would need a prior-shift correction of the intercept and a new cutoff.
- **Wrong code table.** Grömping (2019) also shows that the Statlog code table is wrong for 8
  attributes (checking status, credit history, purpose, savings, personal status/sex, property,
  housing, foreign worker). The model is label-invariant (WoE depends only on the codes), but the
  *meaning* of bins and reason codes changes. This repo keeps the Statlog labels as the API
  vocabulary and adds the corrected meaning to the exported scorecard. A check against her
  published level distributions matches 16 of 17 coded fields to within 0.005 percentage points;
  `housing` differs by 0.14 pp (one applicant).
- **Accepted applicants only.** All 1,000 were granted credit, so the model never sees the
  applicants the bank rejected (no reject inference).

## Features

- **Excluded by design:** `age` and `personal_status` (sex and marital status) are protected
  characteristics in most lending regulation; `foreign_worker` is a national-origin proxy. They
  are still accepted by the API, because monitoring disparities requires them, but the model
  never sees them.
- **Candidates (17):** checking status, duration, credit history, purpose, amount, savings,
  employment, installment rate, other parties, residence, property, other payment plans,
  housing, number of credits, job, dependents, telephone.
- **Kept in the final model (11):** checking status, duration, credit history, purpose, amount,
  savings, employment, installment rate, property, other payment plans, housing.

## Performance

5 x 5 repeated stratified cross-validation; WoE binning, IV filtering and the logistic
regression are refitted inside every training fold.

| Metric | Mean ± SD over 25 folds |
|---|---:|
| ROC-AUC | 0.780 ± 0.029 |
| Gini | 0.559 ± 0.058 |
| PR-AUC (bad = positive) | 0.587 ± 0.045 |
| KS | 0.476 ± 0.054 |
| Brier score | 0.170 ± 0.011 |

- **Calibration** (pooled out-of-fold PDs): ECE 0.029, calibration slope 0.83 (the PDs are
  somewhat too extreme), mean PD 0.298 vs observed bad rate 0.300.
- **Expected cost per applicant:** 0.536 ± 0.010 when the cutoff itself is chosen out of sample
  (0.510 when it is tuned and evaluated on the same predictions). For reference: rejecting
  everyone costs 0.70, approving everyone 1.50, and the default 0.5 cutoff 0.937. About 36% of
  applicants are approved at the deployed cutoff.
- **Versus alternatives:** gradient boosting reaches ROC-AUC 0.783 ± 0.032 and one-hot logistic
  regression 0.781 ± 0.034 (see README); the differences are well inside the fold-to-fold SD.

## Fairness findings

Out-of-fold decisions at the deployed cutoff, averaged over the 5 CV repeats; 95% intervals
from a 2,000-sample bootstrap over applicants. DI = approval-rate ratio (unprivileged /
privileged), EOD = difference in approval rate among applicants who repaid (TPR).

| Comparison | Approval rates | DI [95% CI] | EOD [95% CI] |
|---|---|---:|---:|
| Age < 25 (n=149) vs >= 25 (n=851) | 21.7% vs 38.9% | 0.56 [0.40, 0.73] | -0.175 [-0.274, -0.077] |
| "Female" vs "male", Statlog code table (310 vs 690) | 35.4% vs 36.7% | 0.96 [0.81, 1.14] | +0.024 [-0.048, +0.098] |
| Female vs male, corrected code table (92 vs 598) | 35.2% vs 37.0% | 0.95 [0.69, 1.22] | -0.024 [-0.139, +0.086] |

- **Age:** the scorecard approves applicants under 25 at 0.56 times the rate of older
  applicants, far below the four-fifths (0.8) screening threshold, and creditworthy young
  applicants are approved less often (32.3% vs 49.8%). Age is not a model input; the gap comes
  through correlated features (shorter employment, renting, smaller savings) and reflects a real
  difference in the observed bad rate (40.9% vs 28.1%). Including age as a feature makes it worse
  (DI 0.42). This is not resolved; it is reported.
- **Sex:** no measurable gap under either code table, but the evidence is weak: under the
  corrected codes only 92 applicants are identifiably female and the DI interval spans 0.69-1.22.
  The 310 applicants in Statlog's "female: divorced/separated/married" level are in fact a mix of
  non-single women and single men, so a "sex" column derived from it is wrong.
- **Excluding the attributes matters:** a scorecard that does use them (ablation) has the same
  accuracy (ROC-AUC 0.784 ± 0.027) and cost, but lowers the Statlog "female" DI from 0.96 to 0.77.

## Explanations

Reason codes are exact for this model: the score is additive, so "points lost vs the best bin"
is the feature's full contribution to the gap from the best possible score. Permutation
importance (README) ranks the same top-5 features for the scorecard and for gradient boosting.

## Limitations and ethical considerations

- 1,000 applicants from the 1970s: performance estimates have wide uncertainty (fold SD of
  ROC-AUC 0.03) and the population, products and economy are not current.
- PDs are calibrated to a 30% bad rate that was produced by oversampling; approval rates and
  costs do not transfer to a real portfolio without a prior-shift correction.
- The 5:1 cost matrix is a stylised convention, not a bank's loss-given-default economics.
- Accepted-only data: the model cannot learn about applicants the bank turned down.
- Fairness through unawareness (dropping protected attributes) does not remove the age gap.
  Possible mitigations (group-specific cutoffs, constrained training) trade off against cost and
  can themselves be unlawful disparate treatment; they are not implemented here.
- The four-fifths rule is a screening heuristic from US employment practice, not a legal test
  for credit decisions.
- No drift monitoring (PSI/CSI), no override process and no challenger-model governance.

## How to reproduce

```bash
make install && make run   # writes reports/ and models/scorecard.joblib
```
