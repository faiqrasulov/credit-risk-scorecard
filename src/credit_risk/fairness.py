"""Group fairness audit of accept/reject decisions.

The favourable outcome is *approval* and the "qualified" class is a good credit, so
``TPR = P(approve | good)`` and ``FPR = P(approve | bad)``. Headline measures:

* disparate-impact ratio = approval rate (unprivileged) / approval rate (privileged); the US
  "four-fifths rule" flags values below 0.8;
* equal-opportunity difference = TPR (unprivileged) - TPR (privileged); 0 is parity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from credit_risk.schema import CATEGORICAL_LEVELS

# Statlog code table: "female div/dep/mar" and "female single" are the female levels.
_STATLOG_SEX = {"male div/sep": "male", "female div/dep/mar": "female", "male single": "male",
                "male mar/wid": "male", "female single": "female"}
# Grömping (2019): code A92 holds "female: non-single or male: single" (sex unknown), A93 is
# "male: married/widowed" and A94 is "female: single".
_CORRECTED_SEX = dict(zip(CATEGORICAL_LEVELS["personal_status"],
                          ("male", "mixed", "male", "female", "unknown"), strict=True))


def age_group(age: np.ndarray | pd.Series, cutoff: float = 25) -> np.ndarray:
    """'<25' / '>=25' labels."""
    a = np.asarray(age, dtype="float64")
    return np.where(a < cutoff, f"<{cutoff:g}", f">={cutoff:g}").astype(object)


def sex_statlog(personal_status: np.ndarray | pd.Series) -> np.ndarray:
    """Sex as derived from the (incorrect) Statlog code table, as most fairness papers do."""
    return np.array([_STATLOG_SEX[str(v)] for v in personal_status], dtype=object)


def sex_corrected(personal_status: np.ndarray | pd.Series) -> np.ndarray:
    """Sex under Grömping's corrected code table: 'male', 'female' or 'mixed' (unrecoverable)."""
    return np.array([_CORRECTED_SEX[str(v)] for v in personal_status], dtype=object)


def group_rates(y_true: np.ndarray, approve: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Per-group size, bad rate, approval rate, TPR and FPR.

    ``approve`` may be 0/1 decisions or per-applicant approval frequencies in [0, 1] (e.g. the
    share of repeated-CV runs in which the applicant was approved).
    """
    y = np.asarray(y_true).astype("int64")
    a = np.asarray(approve, dtype="float64")
    g = np.asarray(groups, dtype=object)
    rows = []
    for name in sorted(set(g)):
        m = g == name
        rows.append({"group": name, "n": int(m.sum()), "bad_rate": float(y[m].mean()),
                     "approval_rate": float(a[m].mean()),
                     "tpr": float(a[m & (y == 0)].mean()) if (m & (y == 0)).any() else np.nan,
                     "fpr": float(a[m & (y == 1)].mean()) if (m & (y == 1)).any() else np.nan})
    return pd.DataFrame(rows).set_index("group")


def _ratio_and_gap(y: np.ndarray, a: np.ndarray, unpriv: np.ndarray, priv: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised over rows (bootstrap replicates): DI ratio and equal-opportunity difference."""
    good = y == 0

    def rate(mask: np.ndarray) -> np.ndarray:
        return (a * mask).sum(axis=-1) / np.maximum(mask.sum(axis=-1), 1)

    with np.errstate(divide="ignore", invalid="ignore"):  # empty groups in tiny resamples
        di = rate(unpriv) / rate(priv)
    di = np.where(np.isfinite(di), di, np.nan)
    eod = rate(unpriv & good) - rate(priv & good)
    return di, eod


def fairness_audit(y_true: np.ndarray, approve: np.ndarray, groups: np.ndarray,
                   unprivileged: str, privileged: str, n_boot: int = 2000, seed: int = 0
                   ) -> dict[str, Any]:
    """Group rates, disparate-impact ratio and equal-opportunity difference with bootstrap CIs.

    The bootstrap resamples applicants (percentile 95% intervals), so the intervals reflect
    sampling noise in a 1,000-applicant data set, not just the model.
    """
    y = np.asarray(y_true).astype("int64")
    a = np.asarray(approve, dtype="float64")
    g = np.asarray(groups, dtype=object)
    table = group_rates(y, a, g)
    di, eod = _ratio_and_gap(y, a, g == unprivileged, g == privileged)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    yb, ab, gb = y[idx], a[idx], g[idx]
    di_b, eod_b = _ratio_and_gap(yb, ab, gb == unprivileged, gb == privileged)
    group_ci: dict[str, dict[str, list[float]]] = {}
    for name in table.index:
        m = gb == name
        cis = {}
        for metric, mask in (("approval_rate", m), ("tpr", m & (yb == 0)),
                             ("fpr", m & (yb == 1))):
            rates = (ab * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)
            cis[metric] = [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]
        group_ci[str(name)] = cis
    return {
        "unprivileged": unprivileged,
        "privileged": privileged,
        "groups": {k: {c: float(v) for c, v in row.items()} for k, row in table.iterrows()},
        "groups_ci95": group_ci,
        "disparate_impact": float(di),
        "disparate_impact_ci95": [float(np.nanpercentile(di_b, 2.5)),
                                  float(np.nanpercentile(di_b, 97.5))],
        "equal_opportunity_difference": float(eod),
        "equal_opportunity_difference_ci95": [float(np.nanpercentile(eod_b, 2.5)),
                                              float(np.nanpercentile(eod_b, 97.5))],
    }
