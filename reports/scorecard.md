# Scorecard (final model, fitted on all 1,000 applicants)

Scaling: 600 points at good:bad odds of 50:1, 20 points to double the odds (factor 28.8539, offset 487.1229).
Score = sum of the points of the applicant's bins. Cost-optimal cutoff: reject when PD >= 0.1495, i.e. score <= 537.3.
Points are rounded here; the service uses the unrounded values (`scorecard.csv`).
`Bin` uses the Statlog labels shipped by OpenML; `Meaning` is what the codes mean
according to Grömping (2019), whose corrected code table fixes 8 mislabelled fields.

| Feature | Bin | n | Bad rate | WoE | Points | Meaning (Grömping 2019) |
|---|---|---:|---:|---:|---:|---|
| checking_status | no checking | 394 | 11.7% | +1.171 | 74 | >= 200 DM or salary for at least 1 year |
| checking_status | >=200 | 63 | 22.2% | +0.384 | 55 | 0 <= ... < 200 DM |
| checking_status | 0<=X<200 | 269 | 39.0% | -0.399 | 37 | < 0 DM |
| checking_status | <0 | 274 | 49.3% | -0.814 | 28 | no checking account |
| duration | < 9 | 94 | 10.6% | +1.243 | 74 |  |
| duration | [9, 18) | 339 | 23.6% | +0.328 | 54 |  |
| duration | [18, 36) | 397 | 32.2% | -0.102 | 44 |  |
| duration | [36, 48) | 106 | 43.4% | -0.579 | 34 |  |
| duration | >= 48 | 64 | 56.2% | -1.090 | 23 |  |
| credit_history | critical/other existing credit | 293 | 17.1% | +0.730 | 61 | all credits at this bank paid back duly |
| credit_history | existing paid | 530 | 31.9% | -0.086 | 45 | no credits taken/all credits paid back duly |
| credit_history | delayed previously | 88 | 31.8% | -0.091 | 45 | existing credits paid back duly till now |
| credit_history | all paid \| no credits/all paid | 89 | 59.6% | -1.226 | 22 | critical account/other credits elsewhere \| delay in paying off in the past |
| purpose | retraining \| used car | 112 | 16.1% | +0.789 | 68 | vacation \| car (new) |
| purpose | radio/tv | 280 | 22.1% | +0.410 | 58 | furniture/equipment |
| purpose | furniture/equipment \| domestic appliance | 193 | 32.1% | -0.098 | 44 | car (used) \| radio/television |
| purpose | business \| repairs | 119 | 35.3% | -0.241 | 40 | retraining \| domestic appliances |
| purpose | new car | 234 | 38.0% | -0.356 | 37 | others |
| purpose | other \| education | 62 | 43.5% | -0.586 | 31 | business \| repairs |
| credit_amount | < 709 | 50 | 24.0% | +0.282 | 52 |  |
| credit_amount | [709, 3973) | 700 | 26.1% | +0.194 | 50 |  |
| credit_amount | [3973, 5998) | 100 | 36.0% | -0.273 | 41 |  |
| credit_amount | [5998, 9271) | 100 | 40.0% | -0.441 | 38 |  |
| credit_amount | >= 9271 | 50 | 58.0% | -1.159 | 24 |  |
| savings_status | >=1000 \| no known savings | 231 | 16.5% | +0.771 | 64 | 500 <= ... < 1000 DM \| >= 1000 DM |
| savings_status | 500<=X<1000 | 63 | 17.5% | +0.675 | 61 | 100 <= ... < 500 DM |
| savings_status | 100<=X<500 | 103 | 33.0% | -0.143 | 43 | < 100 DM |
| savings_status | <100 | 603 | 36.0% | -0.269 | 41 | unknown/no savings account |
| employment | 4<=X<7 | 174 | 22.4% | +0.390 | 56 | 4 <= ... < 7 yrs |
| employment | >=7 | 253 | 25.3% | +0.235 | 52 | >= 7 yrs |
| employment | 1<=X<4 | 339 | 30.7% | -0.030 | 46 | 1 <= ... < 4 yrs |
| employment | unemployed | 62 | 37.1% | -0.323 | 39 | unemployed |
| employment | <1 | 172 | 40.7% | -0.468 | 35 | < 1 yr |
| installment_commitment | < 2 | 136 | 25.0% | +0.245 | 57 | >= 35% of income |
| installment_commitment | [2, 3) | 231 | 26.8% | +0.154 | 53 | 25-35% |
| installment_commitment | [3, 4) | 157 | 28.7% | +0.062 | 49 | 20-25% |
| installment_commitment | >= 4 | 476 | 33.4% | -0.155 | 40 | < 20% |
| property_magnitude | real estate | 282 | 21.3% | +0.459 | 49 | unknown/no property |
| property_magnitude | life insurance | 232 | 30.6% | -0.029 | 46 | car or other |
| property_magnitude | car | 332 | 30.7% | -0.033 | 46 | building society savings/life insurance |
| property_magnitude | no known property | 154 | 43.5% | -0.584 | 43 | real estate |
| other_payment_plans | none | 814 | 27.5% | +0.122 | 49 | none |
| other_payment_plans | stores \| bank | 186 | 40.9% | -0.478 | 39 | stores \| bank |
| housing | own | 713 | 26.1% | +0.195 | 49 | rent |
| housing | rent | 179 | 39.1% | -0.404 | 41 | for free |
| housing | for free | 108 | 40.7% | -0.473 | 40 | own |
