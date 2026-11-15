import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from credit_risk.woe import WoEBinner, iv_strength, woe_iv

# Hand-computed example: 3 categories, 80 goods and 60 bads in total.
#   A: 40 good / 10 bad -> %good 0.500, %bad 0.1667, WoE ln(3)     =  1.098612
#   B: 30 good / 20 bad -> %good 0.375, %bad 0.3333, WoE ln(1.125) =  0.117783
#   C: 10 good / 30 bad -> %good 0.125, %bad 0.5000, WoE ln(0.25)  = -1.386294
# IV = 0.3333*1.098612 + 0.0417*0.117783 + (-0.375)*(-1.386294) = 0.890972
HAND_WOE = [1.098612, 0.117783, -1.386294]
HAND_IV = 0.890972


def test_woe_iv_matches_hand_computation():
    woe, iv = woe_iv([40, 30, 10], [10, 20, 30], alpha=0.0)
    assert woe == pytest.approx(HAND_WOE, abs=1e-6)
    assert iv.sum() == pytest.approx(HAND_IV, abs=1e-6)


def test_binner_reproduces_hand_example_end_to_end():
    x = ["A"] * 50 + ["B"] * 50 + ["C"] * 40
    y = [0] * 40 + [1] * 10 + [0] * 30 + [1] * 20 + [0] * 10 + [1] * 30
    b = WoEBinner(alpha=0.0, min_bin_frac=0.0, max_bins=10).fit(pd.DataFrame({"c": x}), y)
    fb = b.binnings_["c"]
    assert fb.labels == ["A", "B", "C"]  # ordered by bad rate
    assert fb.woe == pytest.approx(HAND_WOE, abs=1e-6)
    assert fb.iv == pytest.approx(HAND_IV, abs=1e-6)
    out = b.transform(pd.DataFrame({"c": ["C", "A", "B"]}))
    assert out["c"].to_numpy() == pytest.approx([HAND_WOE[2], HAND_WOE[0], HAND_WOE[1]],
                                                abs=1e-6)


def test_laplace_smoothing_keeps_woe_finite():
    woe, iv = woe_iv([5, 0], [0, 5], alpha=0.5)
    assert np.isfinite(woe).all() and np.isfinite(iv).all()
    assert woe == pytest.approx([math.log(11), -math.log(11)])


def test_numeric_bins_are_monotonic_and_respect_size_limits():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.uniform(0, 100, n)
    y = (rng.random(n) < 0.1 + 0.6 * x / 100).astype(int)
    b = WoEBinner(max_bins=5, min_bin_frac=0.05).fit(pd.DataFrame({"x": x}), y)
    fb = b.binnings_["x"]
    rates = fb.n_bad / (fb.n_good + fb.n_bad)
    assert 2 <= len(fb.labels) <= 5
    assert np.all(np.diff(rates) > 0) and np.all(np.diff(fb.woe) < 0)
    assert (fb.n_good + fb.n_bad).min() >= 0.05 * n


def test_monotonic_constraint_on_u_shaped_risk():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 5000)
    y = (rng.random(5000) < 0.15 + 0.5 * x**2 + 0.1 * x).astype(int)
    fb = WoEBinner(max_bins=6).fit(pd.DataFrame({"x": x}), y).binnings_["x"]
    d = np.diff(fb.woe)
    assert np.all(d > 0) or np.all(d < 0)


def test_unseen_and_missing_values_get_neutral_woe():
    X = pd.DataFrame({"c": ["a", "b"] * 50, "x": np.arange(100.0)})
    y = np.array([0, 1] * 50)
    b = WoEBinner(min_bin_frac=0.0).fit(X, y)
    out = b.transform(pd.DataFrame({"c": ["zzz"], "x": [np.nan]}))
    assert out.iloc[0].tolist() == [0.0, 0.0]
    assert b.bin_indices(pd.DataFrame({"c": ["zzz"], "x": [np.nan]})).iloc[0].tolist() == [-1, -1]


def test_iv_filter_drops_noise_feature():
    rng = np.random.default_rng(2)
    n = 4000
    signal = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-1.5 * signal))).astype(int)
    X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=n)})
    b = WoEBinner(min_iv=0.02).fit(X, y)
    assert b.selected_features_ == ["signal"]
    assert list(b.get_feature_names_out()) == ["signal"]
    table = b.iv_table()
    assert table.iloc[0]["feature"] == "signal" and table.iloc[0]["strength"] != "unpredictive"


def test_sklearn_api_compatibility():
    b = clone(WoEBinner(max_bins=4, alpha=1.0))
    assert b.get_params()["max_bins"] == 4 and b.get_params()["alpha"] == 1.0
    assert iv_strength(0.01) == "unpredictive" and iv_strength(0.6) == "very strong"
