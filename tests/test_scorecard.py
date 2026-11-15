import math

import numpy as np
import pandas as pd
import pytest

from credit_risk.data import MODEL_FEATURES
from credit_risk.scorecard import (
    Scorecard,
    ScorecardLogit,
    ScoreScaler,
    make_scorecard_pipeline,
)


def test_points_scaling_arithmetic():
    s = ScoreScaler(base_score=600, base_odds=50, pdo=20)
    assert s.factor == pytest.approx(20 / math.log(2))
    assert s.offset == pytest.approx(600 - 20 / math.log(2) * math.log(50))
    # PD = 1/(1+odds): 50:1 good:bad odds -> 600 points; doubling the odds adds exactly PDO.
    assert s.score_from_pd(1 / 51) == pytest.approx(600)
    assert s.score_from_pd(1 / 101) == pytest.approx(620)
    assert s.score_from_pd(1 / 26) == pytest.approx(580)
    assert s.score_from_pd(1 / 201) - s.score_from_pd(1 / 101) == pytest.approx(20)
    pds = np.array([0.01, 0.2, 0.5, 0.9])
    assert s.pd_from_score(s.score_from_pd(pds)) == pytest.approx(pds)


@pytest.fixture(scope="module")
def fitted(synthetic):
    X, y = synthetic
    pipe = make_scorecard_pipeline().fit(X[list(MODEL_FEATURES)], y)
    return X, y, Scorecard(pipe)


def test_points_sum_to_the_model_score(fitted):
    X, _, sc = fitted
    pts = sc.points(X)
    assert list(pts.columns) == sc.features
    np.testing.assert_allclose(pts.sum(axis=1), sc.scaler.score_from_pd(sc.predict_pd(X)),
                               atol=1e-8)


def test_points_per_bin_follow_coefficient_times_woe(fitted):
    _, _, sc = fitted
    table = sc.table()
    for _, grp in table.groupby("feature"):
        diff = grp["points"].to_numpy() - grp["points"].to_numpy()[0]
        expected = -sc.scaler.factor * grp["coefficient"].to_numpy()[0] * (
            grp["woe"].to_numpy() - grp["woe"].to_numpy()[0])
        np.testing.assert_allclose(diff, expected, atol=1e-9)
    assert (table["coefficient"] <= 0).all()  # every kept feature has the expected sign


def test_reason_codes_are_the_largest_point_losses(fitted):
    X, _, sc = fitted
    pts = sc.points(X.iloc[:25])
    best = sc.table().groupby("feature")["points"].max()
    for r, codes in enumerate(sc.reason_codes(X.iloc[:25], k=3)):
        lost = (best[sc.features].to_numpy() - pts.iloc[r].to_numpy())
        top = np.sort(lost)[::-1][:3]
        expected = [round(v, 2) for v in top if v > 1e-9]
        assert [c["points_lost"] for c in codes] == pytest.approx(expected, abs=0.011)
        for c in codes:
            assert c["points_lost"] == pytest.approx(c["best_bin_points"] - c["points"],
                                                     abs=0.011)


def test_applicant_in_every_best_bin_gets_no_reason_codes(fitted):
    X, _, sc = fitted
    table = sc.table()
    pts = sc.points(X)
    row = X.iloc[[0]].copy()
    for feat in sc.features:
        best_label = table[table.feature == feat].sort_values("points").iloc[-1]["bin"]
        donor = np.flatnonzero(
            np.isclose(pts[feat].to_numpy(),
                       table[(table.feature == feat) & (table.bin == best_label)]["points"]
                       .iloc[0]))[0]
        row[feat] = X[feat].iloc[donor]
    assert sc.reason_codes(row) == [[]]


def test_wrong_sign_feature_is_dropped():
    rng = np.random.default_rng(3)
    n = 3000
    good_dir = rng.normal(size=n)  # WoE-like: higher = safer
    wrong_dir = rng.normal(size=n)  # higher = riskier, i.e. wrongly oriented
    y = (rng.random(n) < 1 / (1 + np.exp(1.2 * good_dir - 0.8 * wrong_dir))).astype(int)
    m = ScorecardLogit().fit(pd.DataFrame({"good": good_dir, "wrong": wrong_dir}), y)
    assert m.features_ == ["good"] and m.dropped_features_ == ["wrong"]
    assert (m.coef_ < 0).all()
