# Information Value (all 1,000 applicants, descriptive)

Inside cross-validation the binning and the IV >= 0.02 filter are refitted on each training fold; `CV selection rate` is the share of the 25 folds in which the scorecard kept the feature.

| Feature | IV | Strength | Bins | Excluded by policy | CV selection rate |
|---|---:|---|---:|---|---:|
| checking_status | 0.659 | very strong | 4 | no | 100% |
| credit_history | 0.289 | medium | 4 | no | 100% |
| duration | 0.271 | medium | 5 | no | 100% |
| savings_status | 0.187 | medium | 4 | no | 100% |
| purpose | 0.165 | medium | 6 | no | 100% |
| credit_amount | 0.136 | medium | 5 | no | 100% |
| property_magnitude | 0.112 | medium | 4 | no | 100% |
| age | 0.100 | weak | 4 | yes | 0% |
| employment | 0.086 | weak | 5 | no | 100% |
| housing | 0.084 | weak | 3 | no | 100% |
| other_payment_plans | 0.058 | weak | 2 | no | 100% |
| personal_status | 0.045 | weak | 4 | yes | 0% |
| installment_commitment | 0.025 | weak | 4 | no | 76% |
| other_parties | 0.015 | unpredictive | 2 | no | 24% |
| existing_credits | 0.010 | unpredictive | 2 | no | 8% |
| job | 0.009 | unpredictive | 3 | no | 8% |
| own_telephone | 0.006 | unpredictive | 2 | no | 12% |
| residence_since | 0.002 | unpredictive | 2 | no | 0% |
| num_dependents | 0.000 | unpredictive | 2 | no | 0% |
| foreign_worker | 0.000 | unpredictive | 1 | yes | 0% |
