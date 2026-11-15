import numpy as np
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from credit_risk.metrics import (
    calibration_slope_intercept,
    classification_metrics,
    expected_calibration_error,
    ks_statistic,
)


def test_ks_matches_scipy_including_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.normal(size=500) + 0.8 * y, 1)
    assert ks_statistic(y, s) == pytest.approx(ks_2samp(s[y == 1], s[y == 0]).statistic)


def test_gini_and_bundle():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(0.3 + 0.3 * y + rng.normal(0, 0.2, 400), 0.01, 0.99)
    m = classification_metrics(y, p)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, p))
    assert m["gini"] == pytest.approx(2 * m["roc_auc"] - 1)
    assert 0 < m["brier"] < 0.25


def test_ece_zero_when_every_bin_is_calibrated():
    y = np.tile([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], 100)
    assert expected_calibration_error(y, np.full(1000, 0.3)) == pytest.approx(0.0)


def test_calibration_slope_detects_overconfidence():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1.5, 100_000)
    p = 1 / (1 + np.exp(-z))
    y = (rng.random(100_000) < p).astype(int)
    slope, intercept = calibration_slope_intercept(y, p)
    assert slope == pytest.approx(1.0, abs=0.05) and intercept == pytest.approx(0.0, abs=0.05)
    over = 1 / (1 + np.exp(-2 * z))  # twice too extreme
    assert calibration_slope_intercept(y, over)[0] == pytest.approx(0.5, abs=0.03)
