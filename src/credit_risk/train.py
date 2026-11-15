"""Fit the served model: WoE scorecard + cost-optimal cutoff chosen on out-of-fold PDs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

from credit_risk import __version__
from credit_risk.data import EXCLUDED_FEATURES, MODEL_FEATURES
from credit_risk.decision import approval_rate, optimal_threshold
from credit_risk.evaluation import ModelSpec, repeated_cv
from credit_risk.scorecard import Scorecard, ScoreScaler, make_scorecard_pipeline

DEFAULT_MODEL_PATH = Path("models/scorecard.joblib")


def pooled_optimal_threshold(y: np.ndarray, oof: np.ndarray) -> tuple[float, float]:
    """Cost-optimal cutoff over all repeats' out-of-fold PDs (``oof``: n_repeats x n)."""
    yv = np.asarray(y).astype("int64")
    return optimal_threshold(np.tile(yv, oof.shape[0]), oof.ravel())


def train_scorecard(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, n_repeats: int = 5,
                    seed: int = 42, features: tuple[str, ...] = MODEL_FEATURES,
                    scaler: ScoreScaler | None = None, oof: np.ndarray | None = None
                    ) -> dict[str, Any]:
    """Choose the cutoff on repeated-CV OOF PDs, then fit the scorecard on all rows.

    ``oof`` (n_repeats x n) can be passed when the same CV has already been run.
    """
    scaler = scaler or ScoreScaler()
    if oof is None:
        spec = ModelSpec("woe_scorecard", lambda s: make_scorecard_pipeline(), features)
        oof = repeated_cv(X, y, [spec], n_splits=n_splits, n_repeats=n_repeats,
                          seed=seed).oof["woe_scorecard"]
    threshold, cost = pooled_optimal_threshold(np.asarray(y), oof)
    pipeline = make_scorecard_pipeline().fit(X[list(features)], np.asarray(y))
    return {
        "pipeline": pipeline,
        "scaler": scaler,
        "threshold_pd": threshold,
        "cutoff_score": float(scaler.score_from_pd(threshold)),
        "features": list(features),
        "excluded_features": list(EXCLUDED_FEATURES),
        "oof_cost_per_applicant": cost,
        "oof_approval_rate": approval_rate(oof.ravel(), threshold),
        "oof_roc_auc": _mean_auc(np.asarray(y), oof),
        "versions": {"credit_risk": __version__, "scikit-learn": sklearn.__version__},
    }


def _mean_auc(y: np.ndarray, oof: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(y, p) for p in oof]))


def save_artifact(artifact: dict[str, Any], path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_artifact(path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no model at {path}; create it with `credit-risk train`")
    return joblib.load(path)


def scorecard_from_artifact(artifact: dict[str, Any]) -> Scorecard:
    return Scorecard(artifact["pipeline"], artifact["scaler"])
