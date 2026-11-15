"""Cost-sensitive accept/reject decisions.

Decision rule: reject an applicant when PD >= threshold. German Credit's cost matrix (Hofmann,
1994): approving a bad applicant costs 5, rejecting a good one costs 1, correct decisions cost 0.
With calibrated PDs the Bayes-optimal rule rejects when 5 * PD > 1 * (1 - PD), i.e. at
PD >= cost_reject_good / (cost_reject_good + cost_approve_bad) = 1/6.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

COST_APPROVE_BAD = 5.0
COST_REJECT_GOOD = 1.0


def bayes_threshold(cost_approve_bad: float = COST_APPROVE_BAD,
                    cost_reject_good: float = COST_REJECT_GOOD) -> float:
    """PD cutoff that minimises expected cost if the PDs are calibrated."""
    return cost_reject_good / (cost_reject_good + cost_approve_bad)


def cost_curve(y_true: np.ndarray, pd_: np.ndarray, thresholds: np.ndarray,
               cost_approve_bad: float = COST_APPROVE_BAD,
               cost_reject_good: float = COST_REJECT_GOOD) -> np.ndarray:
    """Expected cost per applicant at each threshold (vectorised)."""
    y = np.asarray(y_true).astype(bool)
    p = np.asarray(pd_, dtype="float64")
    t = np.asarray(thresholds, dtype="float64")
    bad, good = np.sort(p[y]), np.sort(p[~y])
    approved_bad = np.searchsorted(bad, t, side="left")  # bads with PD < t
    rejected_good = len(good) - np.searchsorted(good, t, side="left")  # goods with PD >= t
    return (cost_approve_bad * approved_bad + cost_reject_good * rejected_good) / len(p)


def expected_cost(y_true: np.ndarray, pd_: np.ndarray, threshold: float,
                  cost_approve_bad: float = COST_APPROVE_BAD,
                  cost_reject_good: float = COST_REJECT_GOOD) -> float:
    """Expected cost per applicant when rejecting PD >= ``threshold``."""
    return float(cost_curve(y_true, pd_, np.array([threshold]), cost_approve_bad,
                            cost_reject_good)[0])


def approval_rate(pd_: np.ndarray, threshold: float) -> float:
    return float(np.mean(np.asarray(pd_) < threshold))


def optimal_threshold(y_true: np.ndarray, pd_: np.ndarray,
                      cost_approve_bad: float = COST_APPROVE_BAD,
                      cost_reject_good: float = COST_REJECT_GOOD) -> tuple[float, float]:
    """Exact cost-minimising threshold on the given predictions.

    Candidates are the midpoints between consecutive distinct PDs (every distinct decision rule)
    plus "reject all" (0) and "approve all" (just above the largest PD). Returns
    ``(threshold, cost_per_applicant)``; ties go to the lowest threshold (most conservative).
    """
    u = np.unique(np.asarray(pd_, dtype="float64"))
    top = 1.0 if u[-1] < 1.0 else float(np.nextafter(u[-1], np.inf))
    candidates = np.concatenate([[0.0], (u[:-1] + u[1:]) / 2, [top]])
    costs = cost_curve(y_true, pd_, candidates, cost_approve_bad, cost_reject_good)
    best = int(np.argmin(costs))
    return float(candidates[best]), float(costs[best])


def cross_validated_cost(y_true: np.ndarray, pd_: np.ndarray, n_splits: int = 5, seed: int = 0,
                         cost_approve_bad: float = COST_APPROVE_BAD,
                         cost_reject_good: float = COST_REJECT_GOOD) -> float:
    """Cost per applicant when the threshold itself is chosen out of sample.

    The optimal threshold is picked on k-1 folds of the (already out-of-fold) predictions and
    applied to the held-out fold, which removes the optimism of tuning the cutoff in-sample.
    """
    y = np.asarray(y_true).astype("int64")
    p = np.asarray(pd_, dtype="float64")
    total = 0.0
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(p, y):
        t, _ = optimal_threshold(y[tr], p[tr], cost_approve_bad, cost_reject_good)
        total += expected_cost(y[te], p[te], t, cost_approve_bad, cost_reject_good) * len(te)
    return total / len(p)
