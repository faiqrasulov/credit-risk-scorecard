"""Model factories: WoE scorecard, one-hot logistic regression, histogram gradient boosting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from credit_risk.schema import CATEGORICAL_LEVELS

LOGREG_GRID: dict[str, list] = {"logreg__C": [0.01, 0.1, 1.0, 10.0]}
HGB_GRID: dict[str, list] = {
    "hgb__max_iter": [100, 300],
    "hgb__max_leaf_nodes": [7, 15],
    "hgb__min_samples_leaf": [20, 40],
}


def split_columns(features: Sequence[str]) -> tuple[list[str], list[str]]:
    """(categorical, numeric) feature names, preserving order."""
    cat = [f for f in features if f in CATEGORICAL_LEVELS]
    num = [f for f in features if f not in CATEGORICAL_LEVELS]
    return cat, num


def make_onehot_logreg(features: Sequence[str], C: float = 1.0) -> Pipeline:
    """L2 logistic regression on one-hot categoricals and standardised numerics."""
    cat, num = split_columns(features)
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ("num", StandardScaler(), num),
    ])
    return Pipeline([("pre", pre), ("logreg", LogisticRegression(C=C, max_iter=2000))])


def make_hgb(features: Sequence[str], seed: int = 0) -> Pipeline:
    """HistGradientBoosting with native categorical splits (ordinal-encoded categoricals)."""
    cat, num = split_columns(features)
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cat),
        ("num", "passthrough", num),
    ])
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05, l2_regularization=1.0, early_stopping=False,
        categorical_features=[True] * len(cat) + [False] * len(num), random_state=seed)
    return Pipeline([("pre", pre), ("hgb", hgb)])


def tuned(estimator: Pipeline, grid: dict[str, list], seed: int, inner_splits: int = 3
          ) -> GridSearchCV:
    """Inner-CV grid search on log loss (a proper scoring rule for PDs), refit on all rows."""
    return GridSearchCV(estimator, grid, scoring="neg_log_loss", n_jobs=1, refit=True,
                        cv=StratifiedKFold(inner_splits, shuffle=True, random_state=seed),
                        error_score="raise")
