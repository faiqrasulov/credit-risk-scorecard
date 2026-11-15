"""Discrimination and calibration metrics used in credit scoring."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max gap between the score CDFs of bads and goods."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(score, dtype="float64")
    bad, good = np.sort(s[y]), np.sort(s[~y])
    grid = np.unique(s)
    cdf_bad = np.searchsorted(bad, grid, side="right") / len(bad)
    cdf_good = np.searchsorted(good, grid, side="right") / len(good)
    return float(np.max(np.abs(cdf_bad - cdf_good)))


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    """Count-weighted mean |observed bad rate - mean PD| over equal-count bins of PD."""
    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(prob, dtype="float64")
    order = np.argsort(p, kind="stable")
    ece = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            ece += len(chunk) * abs(y[chunk].mean() - p[chunk].mean())
    return float(ece / len(y))


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """ROC-AUC, Gini, PR-AUC (bad = positive), KS, Brier, log loss and ECE of PD predictions."""
    y = np.asarray(y_true).astype("int64")
    p = np.clip(np.asarray(prob, dtype="float64"), 1e-15, 1 - 1e-15)
    auc = roc_auc_score(y, p)
    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
    }


def calibration_slope_intercept(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration ``logit P(y=1) = a + b * logit(p)``; ideal is b = 1, a = 0.

    A slope below 1 means the predictions are too extreme (over-confident).
    """
    p = np.clip(np.asarray(prob, dtype="float64"), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y_true).astype("int64"))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])
