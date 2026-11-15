"""Logistic-regression scorecard on WoE features, scaled to points, with reason codes.

Points scaling (Siddiqi, 2006): with ``factor = PDO / ln 2`` and
``offset = base_score - factor * ln(base_odds)``, the score is

    score = offset + factor * ln(odds_good),   odds_good = (1 - PD) / PD,

so ``base_score`` points correspond to ``base_odds`` good:bad odds and every ``PDO`` extra points
double the odds. Because the model is linear in WoE, the score splits exactly into one points
value per (feature, bin):

    points_j(bin) = (offset - factor * b0) / m - factor * b_j * WoE_j(bin).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from credit_risk.woe import WoEBinner


@dataclass(frozen=True)
class ScoreScaler:
    """Maps PD <-> score points: ``base_score`` at ``base_odds`` (good:bad), +``pdo`` doubles."""

    base_score: float = 600.0
    base_odds: float = 50.0
    pdo: float = 20.0

    @property
    def factor(self) -> float:
        return self.pdo / math.log(2.0)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * math.log(self.base_odds)

    def score_from_pd(self, pd_: np.ndarray | float) -> np.ndarray:
        p = np.clip(np.asarray(pd_, dtype="float64"), 1e-12, 1 - 1e-12)
        return self.offset + self.factor * np.log((1 - p) / p)

    def pd_from_score(self, score: np.ndarray | float) -> np.ndarray:
        odds_good = np.exp((np.asarray(score, dtype="float64") - self.offset) / self.factor)
        return 1.0 / (1.0 + odds_good)


class ScorecardLogit(ClassifierMixin, BaseEstimator):
    """L2 logistic regression on WoE features that drops features with the wrong sign.

    WoE is oriented so that higher = safer, so every coefficient for P(bad) should be negative.
    A positive coefficient (usually collinearity) would hand *more* points to riskier bins; such
    features are removed one at a time, most positive first, and the model is refitted.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, drop_wrong_sign: bool = True):
        self.C = C
        self.max_iter = max_iter
        self.drop_wrong_sign = drop_wrong_sign

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> ScorecardLogit:
        X = pd.DataFrame(X)
        features = [str(c) for c in X.columns]
        X.columns = features
        dropped: list[str] = []
        while True:
            lr = LogisticRegression(C=self.C, max_iter=self.max_iter)
            lr.fit(X[features].to_numpy(dtype="float64"), y)
            coef = lr.coef_[0]
            if not self.drop_wrong_sign or (coef <= 0).all() or len(features) == 1:
                break
            worst = features[int(np.argmax(coef))]
            features.remove(worst)
            dropped.append(worst)
        self.lr_ = lr
        self.features_ = features
        self.dropped_features_ = dropped
        self.classes_ = lr.classes_
        self.coef_ = lr.coef_
        self.intercept_ = lr.intercept_
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "lr_")
        X = pd.DataFrame(X)
        X.columns = [str(c) for c in X.columns]
        return X[self.features_].to_numpy(dtype="float64")

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr_.decision_function(self._design(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.lr_.predict_proba(self._design(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def make_scorecard_pipeline(min_iv: float = 0.02, max_bins: int = 6, min_bin_frac: float = 0.05,
                            alpha: float = 0.5, C: float = 1.0) -> Pipeline:
    """WoE binning (fitted on the training data only) followed by the sign-constrained logit."""
    return Pipeline([
        ("woe", WoEBinner(max_bins=max_bins, min_bin_frac=min_bin_frac, alpha=alpha,
                          min_iv=min_iv)),
        ("logit", ScorecardLogit(C=C)),
    ])


class Scorecard:
    """Points view of a fitted ``make_scorecard_pipeline`` pipeline."""

    def __init__(self, pipeline: Pipeline, scaler: ScoreScaler | None = None):
        self.pipeline = pipeline
        self.scaler = scaler or ScoreScaler()
        self.binner: WoEBinner = pipeline.named_steps["woe"]
        logit: ScorecardLogit = pipeline.named_steps["logit"]
        self.features: list[str] = list(logit.features_)
        coef = dict(zip(logit.features_, logit.coef_[0], strict=True))
        m = len(self.features)
        f = self.scaler.factor
        self.base_points = (self.scaler.offset - f * float(logit.intercept_[0])) / m
        self._points = {
            feat: self.base_points - f * coef[feat] * self.binner.binnings_[feat].woe
            for feat in self.features
        }
        self._coef = coef

    def table(self) -> pd.DataFrame:
        """The scorecard: one row per (feature, bin) with WoE, coefficient and points."""
        rows: list[dict[str, Any]] = []
        for feat in self.features:
            fb = self.binner.binnings_[feat]
            n = fb.n_good + fb.n_bad
            for i, label in enumerate(fb.labels):
                rows.append({"feature": feat, "bin": label, "n": int(n[i]),
                             "bad_rate": float(fb.n_bad[i] / n[i]), "woe": float(fb.woe[i]),
                             "coefficient": float(self._coef[feat]),
                             "points": float(self._points[feat][i])})
        return pd.DataFrame(rows)

    def points(self, X: pd.DataFrame) -> pd.DataFrame:
        """Points of each applicant on each scorecard feature (unseen/missing = WoE 0)."""
        idx = self.binner.bin_indices(X)
        out = {}
        for feat in self.features:
            i = idx[feat].to_numpy()
            out[feat] = np.where(i >= 0, self._points[feat][np.clip(i, 0, None)],
                                 self.base_points)
        return pd.DataFrame(out, index=idx.index)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Total score; equals ``scaler.score_from_pd(predict_pd(X))`` up to rounding."""
        return self.points(X).sum(axis=1).to_numpy()

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def reason_codes(self, X: pd.DataFrame, k: int = 3) -> list[list[dict[str, Any]]]:
        """Adverse-action reasons: the ``k`` features costing the most points vs their best bin."""
        pts = self.points(X)
        idx = self.binner.bin_indices(X)
        best = {feat: float(self._points[feat].max()) for feat in self.features}
        lost = pd.DataFrame({feat: best[feat] - pts[feat] for feat in self.features})
        result: list[list[dict[str, Any]]] = []
        for r in range(len(pts)):
            row_lost = lost.iloc[r]
            order = np.argsort(-row_lost.to_numpy(), kind="stable")[:k]
            codes = []
            for j in order:
                feat = self.features[j]
                if row_lost.iloc[j] <= 1e-9:
                    break
                b = int(idx[feat].iloc[r])
                labels = self.binner.binnings_[feat].labels
                codes.append({
                    "feature": feat,
                    "value": _python_scalar(X[feat].iloc[r]),
                    "bin": labels[b] if b >= 0 else "(unseen)",
                    "points": round(float(pts[feat].iloc[r]), 2),
                    "best_bin_points": round(best[feat], 2),
                    "points_lost": round(float(row_lost.iloc[j]), 2),
                })
            result.append(codes)
        return result


def _python_scalar(v: Any) -> Any:
    """numpy scalar -> plain Python value (JSON-serialisable)."""
    return v.item() if isinstance(v, np.generic) else v
