"""The 20 raw application fields of German Credit (Statlog) and the corrected code table.

Category levels are listed in Statlog code order (A11, A12, ... for attribute A1, etc.), using the
labels OpenML ships. Grömping (2019) showed that the Statlog code table attached to these codes is
wrong for 8 of the 13 categorical attributes; ``CORRECTED_MEANING`` gives, index-aligned with the
levels, the meaning of each code according to her corrected "South German Credit" documentation.
``GROEMPING_TABLE1`` holds her published level distributions (in % of bad and of good credits),
which :func:`credit_risk.data.codebook_check` compares with the data to confirm the alignment.
"""

from __future__ import annotations

FEATURES: tuple[str, ...] = (
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
)

CATEGORICAL_LEVELS: dict[str, tuple[str, ...]] = {
    "checking_status": ("<0", "0<=X<200", ">=200", "no checking"),
    "credit_history": (
        "no credits/all paid",
        "all paid",
        "existing paid",
        "delayed previously",
        "critical/other existing credit",
    ),
    "purpose": (
        "new car",
        "used car",
        "furniture/equipment",
        "radio/tv",
        "domestic appliance",
        "repairs",
        "education",
        "vacation",
        "retraining",
        "business",
        "other",
    ),
    "savings_status": ("<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"),
    "employment": ("unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"),
    "personal_status": (
        "male div/sep",
        "female div/dep/mar",
        "male single",
        "male mar/wid",
        "female single",
    ),
    "other_parties": ("none", "co applicant", "guarantor"),
    "property_magnitude": ("real estate", "life insurance", "car", "no known property"),
    "other_payment_plans": ("bank", "stores", "none"),
    "housing": ("rent", "own", "for free"),
    "job": (
        "unemp/unskilled non res",
        "unskilled resident",
        "skilled",
        "high qualif/self emp/mgmt",
    ),
    "own_telephone": ("none", "yes"),
    "foreign_worker": ("yes", "no"),
}

NUMERIC_FEATURES: tuple[str, ...] = tuple(f for f in FEATURES if f not in CATEGORICAL_LEVELS)
CATEGORICAL_FEATURES: tuple[str, ...] = tuple(f for f in FEATURES if f in CATEGORICAL_LEVELS)

# Integer-coded "numeric" fields: allowed values in Statlog code order.
CODED_NUMERIC_LEVELS: dict[str, tuple[int, ...]] = {
    "installment_commitment": (1, 2, 3, 4),
    "residence_since": (1, 2, 3, 4),
    "existing_credits": (1, 2, 3, 4),
    "num_dependents": (1, 2),
}

# Meaning of each code per Grömping (2019), Table 1, aligned with the levels above.
CORRECTED_MEANING: dict[str, tuple[str, ...]] = {
    "checking_status": (
        "no checking account",
        "< 0 DM",
        "0 <= ... < 200 DM",
        ">= 200 DM or salary for at least 1 year",
    ),
    "credit_history": (
        "delay in paying off in the past",
        "critical account/other credits elsewhere",
        "no credits taken/all credits paid back duly",
        "existing credits paid back duly till now",
        "all credits at this bank paid back duly",
    ),
    "purpose": (
        "others",
        "car (new)",
        "car (used)",
        "furniture/equipment",
        "radio/television",
        "domestic appliances",
        "repairs",
        "education",
        "vacation",
        "retraining",
        "business",
    ),
    "savings_status": (
        "unknown/no savings account",
        "< 100 DM",
        "100 <= ... < 500 DM",
        "500 <= ... < 1000 DM",
        ">= 1000 DM",
    ),
    "employment": ("unemployed", "< 1 yr", "1 <= ... < 4 yrs", "4 <= ... < 7 yrs", ">= 7 yrs"),
    "personal_status": (
        "male: divorced/separated",
        "female: non-single or male: single",
        "male: married/widowed",
        "female: single",
        "(code not used)",
    ),
    "other_parties": ("none", "co-applicant", "guarantor"),
    "property_magnitude": (
        "unknown/no property",
        "car or other",
        "building society savings/life insurance",
        "real estate",
    ),
    "other_payment_plans": ("bank", "stores", "none"),
    "housing": ("for free", "rent", "own"),
    "job": (
        "unemployed/unskilled, non-resident",
        "unskilled, resident",
        "skilled employee/official",
        "manager/self-employed/highly qualified",
    ),
    "own_telephone": ("no", "yes (under customer name)"),
    "foreign_worker": ("no", "yes"),
    "installment_commitment": (">= 35% of income", "25-35%", "20-25%", "< 20%"),
    "residence_since": ("< 1 yr", "1-4 yrs", "4-7 yrs", ">= 7 yrs"),
    "existing_credits": ("1", "2-3", "4-5", ">= 6"),
    "num_dependents": ("0 to 2", "3 or more"),
}

# Grömping (2019), Table 1: % of bad credits and % of good credits per level, in Statlog code
# order (her table lists the corrected codes; levels of num_dependents and foreign_worker are
# swapped relative to Statlog, as her note under the table instructs).
GROEMPING_TABLE1: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "checking_status": ((45.00, 35.00, 4.67, 15.33), (19.86, 23.43, 7.00, 49.71)),
    "credit_history": ((8.33, 9.33, 56.33, 9.33, 16.67), (2.14, 3.00, 51.57, 8.57, 34.71)),
    "purpose": (
        (29.67, 5.67, 19.33, 20.67, 1.33, 2.67, 7.33, 0.00, 0.33, 11.33, 1.67),
        (20.71, 12.29, 17.57, 31.14, 1.14, 2.00, 4.00, 0.00, 1.14, 9.00, 1.00),
    ),
    "savings_status": ((72.33, 11.33, 3.67, 2.00, 10.67), (55.14, 9.86, 7.43, 6.00, 21.57)),
    "employment": ((7.67, 23.33, 34.67, 13.00, 21.33), (5.57, 14.57, 33.57, 19.29, 27.00)),
    "installment_commitment": ((11.33, 20.67, 15.00, 53.00), (14.57, 24.14, 16.00, 45.29)),
    "personal_status": ((6.67, 36.33, 48.67, 8.33, 0.0), (4.29, 28.71, 57.43, 9.57, 0.0)),
    "other_parties": ((90.67, 6.00, 3.33), (90.71, 3.29, 6.00)),
    "residence_since": ((12.00, 32.33, 14.33, 41.33), (13.43, 30.14, 15.14, 41.29)),
    "property_magnitude": ((20.00, 23.67, 34.00, 22.33), (31.71, 23.00, 32.86, 12.43)),
    "other_payment_plans": ((19.00, 6.33, 74.67), (11.71, 4.00, 84.29)),
    "housing": ((23.33, 62.00, 14.67), (15.57, 75.43, 9.00)),
    "existing_credits": ((66.67, 30.67, 2.00, 0.67), (61.86, 34.43, 3.14, 0.57)),
    "job": ((2.33, 18.67, 62.00, 17.00), (2.14, 20.57, 63.43, 13.86)),
    "num_dependents": ((84.67, 15.33), (84.43, 15.57)),
    "own_telephone": ((62.33, 37.67), (58.43, 41.57)),
    "foreign_worker": ((98.67, 1.33), (95.29, 4.71)),
}


def corrected_meaning(feature: str, value: object) -> str | None:
    """Meaning of a raw Statlog value per Grömping (2019), or None for continuous features."""
    if feature in CATEGORICAL_LEVELS:
        levels: tuple = CATEGORICAL_LEVELS[feature]
        key: object = str(value)
    elif feature in CODED_NUMERIC_LEVELS:
        levels = CODED_NUMERIC_LEVELS[feature]
        key = int(float(value))  # type: ignore[arg-type]
    else:
        return None
    if key not in levels:
        return None
    return CORRECTED_MEANING[feature][levels.index(key)]
