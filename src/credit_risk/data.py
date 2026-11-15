"""Loading German Credit (Statlog), model feature lists and a synthetic look-alike for tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk.schema import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_LEVELS,
    CODED_NUMERIC_LEVELS,
    FEATURES,
    GROEMPING_TABLE1,
)

OPENML_DATA_ID = 31  # "credit-g", the Statlog German Credit data (Hofmann, 1994)

# Attributes kept out of every model: age and personal_status (which Statlog documents as sex +
# marital status) are protected characteristics in most lending regulation; foreign_worker is a
# national-origin proxy. They are still loaded, because the fairness audit needs them.
EXCLUDED_FEATURES: tuple[str, ...] = ("age", "personal_status", "foreign_worker")
MODEL_FEATURES: tuple[str, ...] = tuple(f for f in FEATURES if f not in EXCLUDED_FEATURES)


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 20 raw fields with plain dtypes: float64 numerics and object-dtype strings."""
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    out = pd.DataFrame(index=df.index)
    for f in FEATURES:
        if f in CATEGORICAL_LEVELS:
            out[f] = pd.Series(df[f].astype(str).to_numpy(dtype=object), index=df.index,
                               dtype=object)
        else:
            out[f] = pd.to_numeric(df[f]).astype("float64")
    return out


def load_german_credit(data_home: str | Path = "data") -> tuple[pd.DataFrame, pd.Series]:
    """Download (once, then from cache) German Credit from OpenML.

    Returns the 20 raw features and the target ``bad`` (1 = bad credit risk, 0 = good).
    """
    from sklearn.datasets import fetch_openml

    Path(data_home).mkdir(parents=True, exist_ok=True)
    try:
        bunch = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, data_home=str(data_home))
    except Exception as exc:  # network errors surface as several exception types
        raise RuntimeError(
            "Could not load German Credit (OpenML data_id=31). The first run needs internet "
            f"access; afterwards the copy cached in '{data_home}/' is used. Cause: {exc!r}"
        ) from exc
    X = clean_frame(bunch.data)
    target = bunch.target.astype(str)
    if set(target.unique()) != {"good", "bad"}:
        raise ValueError(f"unexpected target levels: {sorted(target.unique())}")
    y = pd.Series((target == "bad").astype("int64").to_numpy(), index=X.index, name="bad")
    validate_levels(X)
    return X, y


def validate_levels(X: pd.DataFrame) -> None:
    """Raise if a categorical column holds a level outside the documented Statlog code table."""
    for f in CATEGORICAL_FEATURES:
        unknown = set(X[f].unique()) - set(CATEGORICAL_LEVELS[f])
        if unknown:
            raise ValueError(f"{f}: unknown levels {sorted(unknown)}")


def codebook_check(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Max absolute gap (percentage points) between the data and Grömping's Table 1, per field.

    Small gaps (rounding, < 0.01 pp) confirm that the corrected code table in
    :mod:`credit_risk.schema` is aligned with the Statlog codes as OpenML ships them.
    """
    yv = np.asarray(y)
    gaps: dict[str, float] = {}
    for f, (bad_pct, good_pct) in GROEMPING_TABLE1.items():
        levels = CATEGORICAL_LEVELS.get(f) or CODED_NUMERIC_LEVELS[f]
        col = X[f].to_numpy()
        gap = 0.0
        for k, level in enumerate(levels):
            hit = col == level
            obs_bad = 100.0 * hit[yv == 1].mean()
            obs_good = 100.0 * hit[yv == 0].mean()
            gap = max(gap, abs(obs_bad - bad_pct[k]), abs(obs_good - good_pct[k]))
        gaps[f] = round(gap, 4)
    return gaps


def make_synthetic_credit(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic applicants with the German Credit schema (for tests; no real data).

    Risk is driven by a known logistic model on checking status, duration, amount, savings and
    credit history, so a working pipeline must recover a clearly better-than-chance ranking.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for f in CATEGORICAL_FEATURES:
        levels = [lv for lv in CATEGORICAL_LEVELS[f] if lv not in ("vacation", "female single")]
        data[f] = rng.choice(np.array(levels, dtype=object), size=n)
    for f, levels in CODED_NUMERIC_LEVELS.items():
        data[f] = rng.choice(np.array(levels, dtype="float64"), size=n)
    data["duration"] = rng.integers(4, 73, size=n).astype("float64")
    data["credit_amount"] = np.round(rng.lognormal(7.8, 0.75, size=n)).clip(250, 20000)
    data["age"] = rng.integers(19, 76, size=n).astype("float64")
    X = pd.DataFrame({f: data[f] for f in FEATURES})
    for f in CATEGORICAL_FEATURES:
        X[f] = X[f].astype(object)
    logit = (
        -1.0
        + 1.2 * (X["checking_status"] == "<0")
        - 1.0 * (X["checking_status"] == "no checking")
        + 0.035 * (X["duration"] - 20)
        + 0.25 * np.log(X["credit_amount"] / 2500)
        - 0.6 * (X["savings_status"] == "no known savings")
        + 0.8 * (X["credit_history"] == "no credits/all paid")
    ).to_numpy(dtype="float64")
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int64"), name="bad")
    return X, y
