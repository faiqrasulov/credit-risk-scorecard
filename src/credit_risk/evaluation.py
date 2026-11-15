"""Repeated stratified cross-validation with out-of-fold predictions for every model."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold

from credit_risk.metrics import classification_metrics

METRICS = ("roc_auc", "gini", "pr_auc", "ks", "brier", "log_loss", "ece")


@dataclass
class ModelSpec:
    """A model to evaluate: ``build(seed)`` returns an unfitted estimator."""

    name: str
    build: Callable[[int], Any]
    features: Sequence[str]
    calibrate: Sequence[str] = ()  # CalibratedClassifierCV methods applied to the fitted model


@dataclass
class CVResult:
    oof: dict[str, np.ndarray]  # model -> (n_repeats, n) out-of-fold PDs
    folds: pd.DataFrame  # one row per (model, repeat, fold) with all metrics
    best_params: dict[str, list[dict]] = field(default_factory=dict)
    importance: dict[str, pd.DataFrame] = field(default_factory=dict)
    scorecard_features: dict[str, list[list[str]]] = field(default_factory=dict)

    def summary(self) -> dict[str, dict[str, dict[str, float]]]:
        """Mean and standard deviation over folds of every metric, per model."""
        out: dict[str, dict[str, dict[str, float]]] = {}
        for name, df in self.folds.groupby("model", sort=False):
            out[str(name)] = {m: {"mean": float(df[m].mean()), "std": float(df[m].std(ddof=1))}
                              for m in METRICS}
        return out


def repeated_cv(X: pd.DataFrame, y: pd.Series, specs: Sequence[ModelSpec], n_splits: int = 5,
                n_repeats: int = 5, seed: int = 42, importance_for: Sequence[str] = (),
                importance_repeats: int = 10, log: Callable[[str], None] | None = None
                ) -> CVResult:
    """Fit every spec on every training fold; collect OOF PDs and per-fold metrics.

    All preprocessing (WoE binning, encoders, tuning, calibration) lives inside the estimators,
    so it is refitted on each training fold and never sees the test fold. Permutation importance
    (ROC-AUC drop) is computed on the test folds of the first repeat.
    """
    yv = np.asarray(y).astype("int64")
    n = len(yv)
    oof: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    best_params: dict[str, list[dict]] = {}
    importance_parts: dict[str, list[pd.DataFrame]] = {}
    sc_features: dict[str, list[list[str]]] = {}
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for k, (tr, te) in enumerate(cv.split(np.zeros(n), yv)):
        r, f = divmod(k, n_splits)
        for spec in specs:
            cols = list(spec.features)
            Xtr, Xte = X.iloc[tr][cols], X.iloc[te][cols]
            est = spec.build(seed + k)
            est.fit(Xtr, yv[tr])
            fitted = est.best_estimator_ if isinstance(est, GridSearchCV) else est
            if isinstance(est, GridSearchCV):
                best_params.setdefault(spec.name, []).append(est.best_params_)
            if hasattr(fitted, "named_steps") and "logit" in fitted.named_steps:
                sc_features.setdefault(spec.name, []).append(
                    list(fitted.named_steps["logit"].features_))
            preds = {spec.name: est.predict_proba(Xte)[:, 1]}
            for method in spec.calibrate:
                cal = CalibratedClassifierCV(
                    clone(fitted), method=method,
                    cv=StratifiedKFold(5, shuffle=True, random_state=seed + k))
                cal.fit(Xtr, yv[tr])
                preds[f"{spec.name}_{method}"] = cal.predict_proba(Xte)[:, 1]
            if r == 0 and spec.name in importance_for:
                pi = permutation_importance(fitted, Xte, yv[te], scoring="roc_auc",
                                            n_repeats=importance_repeats, random_state=seed + k)
                importance_parts.setdefault(spec.name, []).append(
                    pd.DataFrame({"feature": cols, "importance": pi.importances_mean}))
            for name, p in preds.items():
                oof.setdefault(name, np.full((n_repeats, n), np.nan))[r, te] = p
                rows.append({"model": name, "repeat": r, "fold": f,
                             **classification_metrics(yv[te], p)})
        if log is not None:
            log(f"fold {k + 1}/{n_splits * n_repeats} done")
    importance = {
        name: (pd.concat(parts).groupby("feature")["importance"].agg(["mean", "std"])
               .sort_values("mean", ascending=False))
        for name, parts in importance_parts.items()
    }
    return CVResult(oof=oof, folds=pd.DataFrame(rows), best_params=best_params,
                    importance=importance, scorecard_features=sc_features)
