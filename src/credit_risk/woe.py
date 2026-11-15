"""Weight-of-Evidence (WoE) binning and Information Value (IV), implemented from scratch.

Conventions (Siddiqi, 2006): the target is 1 = bad, and for bin i

    WoE_i = ln( %good_i / %bad_i ),   IV = sum_i (%good_i - %bad_i) * WoE_i,

so a positive WoE marks a lower-risk bin. Laplace smoothing adds ``alpha`` to every bin's good
and bad count, which keeps WoE finite for bins without goods or without bads.

Numeric features: quantile pre-bins -> merge bins smaller than ``min_bin_frac`` -> pool adjacent
violators until the bad rate is strictly monotonic (both directions tried, the higher-IV one kept)
-> merge the adjacent pair that loses the least IV until at most ``max_bins`` remain.
Categorical features: levels sorted by bad rate -> rare levels merged into the neighbour with the
closest bad rate -> least-IV-loss merging down to ``max_bins``. Unseen levels (and missing
values) get WoE 0, i.e. the population average.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def woe_iv(n_good: Sequence[float], n_bad: Sequence[float], alpha: float = 0.5
           ) -> tuple[np.ndarray, np.ndarray]:
    """WoE per bin and each bin's IV contribution, with Laplace smoothing ``alpha``."""
    g = np.asarray(n_good, dtype="float64")
    b = np.asarray(n_bad, dtype="float64")
    k = len(g)
    dist_good = (g + alpha) / (g.sum() + alpha * k)
    dist_bad = (b + alpha) / (b.sum() + alpha * k)
    woe = np.log(dist_good / dist_bad)
    return woe, (dist_good - dist_bad) * woe


def iv_strength(iv: float) -> str:
    """Rule-of-thumb label for an IV value (Siddiqi, 2006)."""
    if iv < 0.02:
        return "unpredictive"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "very strong"


@dataclass
class _Bins:
    """Ordered bins under construction; ``parts`` holds pre-bin ids or category levels."""

    parts: list[list]
    good: list[float]
    bad: list[float]

    def __len__(self) -> int:
        return len(self.good)

    def copy(self) -> _Bins:
        return _Bins([list(p) for p in self.parts], list(self.good), list(self.bad))

    def merge(self, i: int) -> None:
        """Merge bin ``i`` with bin ``i + 1``."""
        self.parts[i] += self.parts.pop(i + 1)
        self.good[i] += self.good.pop(i + 1)
        self.bad[i] += self.bad.pop(i + 1)

    def bad_rate(self, i: int) -> float:
        return self.bad[i] / (self.good[i] + self.bad[i])

    def iv(self, alpha: float) -> float:
        return float(woe_iv(self.good, self.bad, alpha)[1].sum())


def _merge_small(bins: _Bins, min_count: int) -> None:
    """Merge the smallest under-sized bin into the neighbour with the closer bad rate."""
    while len(bins) > 1:
        sizes = [g + b for g, b in zip(bins.good, bins.bad, strict=True)]
        small = [i for i, s in enumerate(sizes) if s < min_count]
        if not small:
            return
        i = min(small, key=lambda j: sizes[j])
        if i == 0:
            bins.merge(0)
        elif i == len(bins) - 1:
            bins.merge(i - 1)
        else:
            r = bins.bad_rate(i)
            left_gap = abs(bins.bad_rate(i - 1) - r)
            right_gap = abs(bins.bad_rate(i + 1) - r)
            bins.merge(i - 1 if left_gap <= right_gap else i)


def _pool_adjacent_violators(bins: _Bins, increasing: bool) -> None:
    """Merge adjacent bins until the bad rate is strictly monotonic in the given direction."""
    i = 0
    while i < len(bins) - 1:
        r0, r1 = bins.bad_rate(i), bins.bad_rate(i + 1)
        if (r1 <= r0) if increasing else (r1 >= r0):
            bins.merge(i)
            i = max(i - 1, 0)
        else:
            i += 1


def _reduce_to(bins: _Bins, max_bins: int, alpha: float) -> None:
    """Greedily merge the adjacent pair whose merge loses the least IV."""
    while len(bins) > max_bins:
        losses = []
        for i in range(len(bins) - 1):
            trial = bins.copy()
            trial.merge(i)
            losses.append(-trial.iv(alpha))
        bins.merge(int(np.argmin(losses)))


@dataclass
class FeatureBinning:
    """Fitted bins of one feature."""

    feature: str
    kind: str  # "numeric" or "categorical"
    labels: list[str]
    n_good: np.ndarray
    n_bad: np.ndarray
    woe: np.ndarray
    iv: float
    alpha: float = 0.5
    edges: np.ndarray = field(default_factory=lambda: np.empty(0))
    categories: dict[str, int] = field(default_factory=dict)

    def bin_index(self, values: Iterable) -> np.ndarray:
        """Bin of each value; -1 for missing values and unseen categories."""
        if self.kind == "numeric":
            v = np.asarray(values, dtype="float64")
            idx = np.searchsorted(self.edges, v, side="right").astype("int64")
            idx[np.isnan(v)] = -1
            return idx
        vals = list(values)
        return np.fromiter((self.categories.get(str(v), -1) for v in vals), dtype="int64",
                           count=len(vals))

    def woe_of(self, values: Iterable) -> np.ndarray:
        """WoE of each value (0.0, the population average, for bin -1)."""
        idx = self.bin_index(values)
        return np.where(idx >= 0, self.woe[np.clip(idx, 0, None)], 0.0)

    def table(self) -> pd.DataFrame:
        """One row per bin: counts, bad rate, WoE and IV contribution."""
        n = self.n_good + self.n_bad
        _, iv_parts = woe_iv(self.n_good, self.n_bad, self.alpha)
        return pd.DataFrame({
            "feature": self.feature,
            "bin": self.labels,
            "n": n.astype("int64"),
            "n_bad": self.n_bad.astype("int64"),
            "bad_rate": self.n_bad / n,
            "woe": self.woe,
            "iv_contribution": iv_parts,
        })


def _fmt(v: float) -> str:
    return f"{v:g}"


def _numeric_labels(edges: np.ndarray) -> list[str]:
    if len(edges) == 0:
        return ["all values"]
    labels = [f"< {_fmt(edges[0])}"]
    labels += [f"[{_fmt(a)}, {_fmt(b)})" for a, b in zip(edges[:-1], edges[1:], strict=True)]
    labels.append(f">= {_fmt(edges[-1])}")
    return labels


def _prebin_edges(x: np.ndarray, max_prebins: int) -> np.ndarray:
    """Inner cut points; bins are ``x < e0``, ``e0 <= x < e1``, ..., all non-empty."""
    u = np.unique(x)
    if len(u) <= max_prebins:
        return u[1:]
    xs = np.sort(x)
    pos = (np.arange(1, max_prebins) * len(xs)) // max_prebins
    edges = np.unique(xs[pos])
    return edges[edges > xs[0]]


def _finalise(feature: str, kind: str, bins: _Bins, alpha: float, labels: list[str],
              **kw) -> FeatureBinning:
    woe, iv_parts = woe_iv(bins.good, bins.bad, alpha)
    return FeatureBinning(feature=feature, kind=kind, labels=labels,
                          n_good=np.asarray(bins.good, dtype="float64"),
                          n_bad=np.asarray(bins.bad, dtype="float64"),
                          woe=woe, iv=float(iv_parts.sum()), alpha=alpha, **kw)


def fit_numeric_binning(feature: str, x: np.ndarray, y: np.ndarray, *, max_bins: int = 6,
                        max_prebins: int = 20, min_count: int = 1, alpha: float = 0.5,
                        monotonic: bool = True) -> FeatureBinning:
    """Monotonic WoE binning of a numeric feature (missing values are ignored when fitting)."""
    ok = ~np.isnan(x)
    x, y = x[ok], y[ok]
    edges0 = _prebin_edges(x, max_prebins)
    idx = np.searchsorted(edges0, x, side="right")
    k0 = len(edges0) + 1
    good = np.bincount(idx[y == 0], minlength=k0).astype("float64")
    bad = np.bincount(idx[y == 1], minlength=k0).astype("float64")
    bins = _Bins([[j] for j in range(k0)], list(good), list(bad))
    _merge_small(bins, min_count)
    if monotonic and len(bins) > 1:
        candidates = []
        for increasing in (True, False):
            trial = bins.copy()
            _pool_adjacent_violators(trial, increasing)
            _reduce_to(trial, max_bins, alpha)
            candidates.append(trial)
        bins = max(candidates, key=lambda b: b.iv(alpha))
    else:
        _reduce_to(bins, max_bins, alpha)
    edges = np.asarray([edges0[p[0] - 1] for p in bins.parts[1:]], dtype="float64")
    return _finalise(feature, "numeric", bins, alpha, _numeric_labels(edges), edges=edges)


def fit_categorical_binning(feature: str, x: np.ndarray, y: np.ndarray, *, max_bins: int = 6,
                            min_count: int = 1, alpha: float = 0.5) -> FeatureBinning:
    """WoE grouping of a categorical feature: levels ordered by bad rate, then merged."""
    xs = np.asarray([str(v) for v in x], dtype=object)
    cats, inv = np.unique(xs, return_inverse=True)
    good = np.bincount(inv[y == 0], minlength=len(cats)).astype("float64")
    bad = np.bincount(inv[y == 1], minlength=len(cats)).astype("float64")
    rate = (bad + alpha) / (good + bad + 2 * alpha)
    order = np.argsort(rate, kind="stable")
    bins = _Bins([[str(cats[o])] for o in order], list(good[order]), list(bad[order]))
    _merge_small(bins, min_count)
    _reduce_to(bins, max_bins, alpha)
    mapping = {c: i for i, part in enumerate(bins.parts) for c in part}
    labels = [" | ".join(part) for part in bins.parts]
    return _finalise(feature, "categorical", bins, alpha, labels, categories=mapping)


class WoEBinner(TransformerMixin, BaseEstimator):
    """Scikit-learn transformer: raw features -> WoE values, with IV-based feature selection.

    Parameters
    ----------
    categorical_features : columns to treat as categorical (default: all non-numeric columns).
    max_bins : maximum number of bins per feature after merging.
    max_prebins : number of quantile pre-bins for numeric features.
    min_bin_frac : minimum share of rows per bin.
    alpha : Laplace smoothing added to each bin's good and bad counts.
    monotonic : force a monotonic bad rate across numeric bins.
    min_iv : features with IV below this are dropped from the output (0.02 = "unpredictive").
    """

    def __init__(self, categorical_features: Sequence[str] | None = None, max_bins: int = 6,
                 max_prebins: int = 20, min_bin_frac: float = 0.05, alpha: float = 0.5,
                 monotonic: bool = True, min_iv: float = 0.0):
        self.categorical_features = categorical_features
        self.max_bins = max_bins
        self.max_prebins = max_prebins
        self.min_bin_frac = min_bin_frac
        self.alpha = alpha
        self.monotonic = monotonic
        self.min_iv = min_iv

    def fit(self, X: pd.DataFrame, y: Sequence[int]) -> WoEBinner:
        X = _as_frame(X)
        yv = np.asarray(y).astype("int64")
        if len(X) != len(yv):
            raise ValueError("X and y have different lengths")
        if not set(np.unique(yv)) <= {0, 1}:
            raise ValueError("y must be binary 0/1 (1 = bad)")
        if self.categorical_features is None:
            cat = {c for c in X.columns if not is_numeric_dtype(X[c])}
        else:
            cat = set(self.categorical_features)
        min_count = max(1, math.ceil(self.min_bin_frac * len(X)))
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        self.binnings_: dict[str, FeatureBinning] = {}
        for c in X.columns:
            if c in cat:
                self.binnings_[c] = fit_categorical_binning(
                    c, X[c].to_numpy(), yv, max_bins=self.max_bins, min_count=min_count,
                    alpha=self.alpha)
            else:
                self.binnings_[c] = fit_numeric_binning(
                    c, X[c].to_numpy(dtype="float64"), yv, max_bins=self.max_bins,
                    max_prebins=self.max_prebins, min_count=min_count, alpha=self.alpha,
                    monotonic=self.monotonic)
        ivs = pd.Series({c: b.iv for c, b in self.binnings_.items()}, name="iv")
        selected = [c for c in X.columns if self.binnings_[c].iv >= self.min_iv]
        self.selected_features_: list[str] = selected or [str(ivs.idxmax())]
        self.iv_ = ivs.sort_values(ascending=False)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "binnings_")
        X = _as_frame(X, self.feature_names_in_)
        return pd.DataFrame(
            {c: self.binnings_[c].woe_of(X[c].to_numpy()) for c in self.selected_features_},
            index=X.index,
        )

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        check_is_fitted(self, "binnings_")
        return np.asarray(self.selected_features_, dtype=object)

    def bin_indices(self, X: pd.DataFrame) -> pd.DataFrame:
        """Bin index (-1 = unseen/missing) of each selected feature."""
        check_is_fitted(self, "binnings_")
        X = _as_frame(X, self.feature_names_in_)
        return pd.DataFrame(
            {c: self.binnings_[c].bin_index(X[c].to_numpy()) for c in self.selected_features_},
            index=X.index,
        )

    def iv_table(self) -> pd.DataFrame:
        """IV of every input feature, strongest first, with its selection status."""
        check_is_fitted(self, "binnings_")
        rows = [
            {"feature": f, "iv": iv, "n_bins": len(self.binnings_[f].labels),
             "strength": iv_strength(iv), "selected": f in self.selected_features_}
            for f, iv in self.iv_.items()
        ]
        return pd.DataFrame(rows)

    def binning_table(self) -> pd.DataFrame:
        """Bin-level table of every input feature."""
        check_is_fitted(self, "binnings_")
        return pd.concat([b.table() for b in self.binnings_.values()], ignore_index=True)


def _as_frame(X, columns: Sequence[str] | None = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        if columns is not None:
            missing = [c for c in columns if c not in X.columns]
            if missing:
                raise ValueError(f"missing columns: {missing}")
            return X.loc[:, list(columns)]
        return X
    arr = np.asarray(X, dtype=object)
    names = list(columns) if columns is not None else [f"x{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=names)
