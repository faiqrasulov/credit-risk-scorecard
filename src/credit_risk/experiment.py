"""Full reproduction: model comparison, calibration, cost analysis, fairness audit, artifacts.

Everything in ``reports/metrics.json`` and ``reports/figures`` is produced by
:func:`run_experiment`.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.data import EXCLUDED_FEATURES, MODEL_FEATURES, codebook_check
from credit_risk.decision import (
    COST_APPROVE_BAD,
    COST_REJECT_GOOD,
    approval_rate,
    bayes_threshold,
    cross_validated_cost,
    expected_cost,
    optimal_threshold,
)
from credit_risk.evaluation import CVResult, ModelSpec, repeated_cv
from credit_risk.fairness import age_group, fairness_audit, sex_corrected, sex_statlog
from credit_risk.metrics import calibration_slope_intercept, classification_metrics
from credit_risk.models import HGB_GRID, LOGREG_GRID, make_hgb, make_onehot_logreg, tuned
from credit_risk.schema import CODED_NUMERIC_LEVELS, FEATURES, corrected_meaning
from credit_risk.scorecard import Scorecard, make_scorecard_pipeline
from credit_risk.train import pooled_optimal_threshold, save_artifact, train_scorecard
from credit_risk.woe import WoEBinner

SERVED_MODEL = "woe_scorecard"
ABLATION_MODEL = "woe_scorecard_with_protected"
MODEL_LABELS = {
    "woe_scorecard": "WoE scorecard",
    "logreg_onehot": "Logistic regression (one-hot)",
    "hgb": "Gradient boosting (HGB)",
    "hgb_sigmoid": "HGB + sigmoid calibration",
    "hgb_isotonic": "HGB + isotonic calibration",
    ABLATION_MODEL: "WoE scorecard incl. age/sex/foreign",
}


def _ms(values: list[float] | np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype="float64")
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}


def build_specs(hgb_grid: dict[str, list] | None = None,
                logreg_grid: dict[str, list] | None = None) -> list[ModelSpec]:
    """The compared models; all use the same 17 features except the protected-attribute
    ablation."""
    hgb_grid = hgb_grid or HGB_GRID
    lr_grid = logreg_grid or LOGREG_GRID
    feats = MODEL_FEATURES
    return [
        ModelSpec(SERVED_MODEL, lambda s: make_scorecard_pipeline(), feats),
        ModelSpec("logreg_onehot", lambda s: tuned(make_onehot_logreg(feats), lr_grid, s), feats),
        ModelSpec("hgb", lambda s: tuned(make_hgb(feats, s), hgb_grid, s), feats,
                  calibrate=("sigmoid", "isotonic")),
        ModelSpec(ABLATION_MODEL, lambda s: make_scorecard_pipeline(), FEATURES),
    ]


def cost_analysis(y: np.ndarray, oof: np.ndarray, seed: int) -> dict[str, Any]:
    """Expected cost per applicant at 0.5, at the Bayes cutoff 1/6 and at the optimal cutoff."""
    t_bayes = bayes_threshold()
    res: dict[str, list[float]] = {k: [] for k in (
        "c05", "a05", "cb", "ab", "t_opt", "c_opt", "a_opt", "c_cv")}
    for r, p in enumerate(oof):
        res["c05"].append(expected_cost(y, p, 0.5))
        res["a05"].append(approval_rate(p, 0.5))
        res["cb"].append(expected_cost(y, p, t_bayes))
        res["ab"].append(approval_rate(p, t_bayes))
        t, c = optimal_threshold(y, p)
        res["t_opt"].append(t)
        res["c_opt"].append(c)
        res["a_opt"].append(approval_rate(p, t))
        res["c_cv"].append(cross_validated_cost(y, p, seed=seed + r))
    t_pool, c_pool = pooled_optimal_threshold(y, oof)
    return {
        "threshold_0.5": {"cost": _ms(res["c05"]), "approval_rate": _ms(res["a05"])},
        "bayes_threshold_1/6": {"threshold": t_bayes, "cost": _ms(res["cb"]),
                                "approval_rate": _ms(res["ab"])},
        "optimal_in_sample": {"threshold": _ms(res["t_opt"]), "cost": _ms(res["c_opt"]),
                              "approval_rate": _ms(res["a_opt"])},
        "optimal_threshold_cross_validated": {"cost": _ms(res["c_cv"]),
                                              "cost_per_repeat": res["c_cv"]},
        "pooled_optimal_threshold": t_pool,
        "pooled_optimal_cost": c_pool,
        "pooled_optimal_approval_rate": approval_rate(oof.ravel(), t_pool),
    }


def calibration_summary(y: np.ndarray, oof: np.ndarray) -> dict[str, float]:
    """Calibration of the pooled OOF PDs of all repeats."""
    yy = np.tile(y, oof.shape[0])
    m = classification_metrics(yy, oof.ravel())
    slope, intercept = calibration_slope_intercept(yy, oof.ravel())
    return {"brier": m["brier"], "ece": m["ece"], "log_loss": m["log_loss"],
            "calibration_slope": slope, "calibration_intercept": intercept,
            "mean_pd": float(oof.mean()), "observed_bad_rate": float(np.mean(y))}


def fairness_section(X: pd.DataFrame, y: np.ndarray, oof: np.ndarray, threshold: float,
                     n_boot: int, seed: int) -> dict[str, Any]:
    """Audit by age and sex at the deployed cutoff; approvals averaged over CV repeats."""
    approve = (oof < threshold).mean(axis=0)
    audits = {
        "age": (age_group(X["age"]), "<25", ">=25"),
        "sex_statlog_codes": (sex_statlog(X["personal_status"]), "female", "male"),
        "sex_corrected_codes": (sex_corrected(X["personal_status"]), "female", "male"),
    }
    return {name: fairness_audit(y, approve, g, unpriv, priv, n_boot=n_boot, seed=seed)
            for name, (g, unpriv, priv) in audits.items()}


def bin_meaning(feature: str, scorecard: Scorecard, bin_idx: int) -> str:
    """Corrected meaning (Grömping 2019) of the raw codes inside one scorecard bin."""
    fb = scorecard.binner.binnings_[feature]
    if fb.kind == "categorical":
        levels = [lv for lv, b in fb.categories.items() if b == bin_idx]
    elif feature in CODED_NUMERIC_LEVELS:
        codes = np.asarray(CODED_NUMERIC_LEVELS[feature], dtype="float64")
        levels = [int(c) for c, b in zip(codes, fb.bin_index(codes), strict=True) if b == bin_idx]
    else:
        return ""
    return " | ".join(str(corrected_meaning(feature, lv)) for lv in levels)


def scorecard_report(scorecard: Scorecard) -> pd.DataFrame:
    """The exported scorecard: (feature, bin, points) plus counts, WoE and corrected meaning."""
    table = scorecard.table()
    meanings = []
    for feat, grp in table.groupby("feature", sort=False):
        meanings += [bin_meaning(str(feat), scorecard, i) for i in range(len(grp))]
    table["meaning_groemping_2019"] = meanings
    return table


def _write_scorecard(table: pd.DataFrame, reports: Path, scorecard: Scorecard,
                     threshold: float) -> None:
    n_rows = int(table.groupby("feature")["n"].sum().iloc[0])
    out = table.copy()
    out["bad_rate"] = out["bad_rate"].round(4)
    out["woe"] = out["woe"].round(4)
    out["coefficient"] = out["coefficient"].round(4)
    out["points"] = out["points"].round(2)
    out.to_csv(reports / "scorecard.csv", index=False)
    s = scorecard.scaler
    lines = [
        f"# Scorecard (final model, fitted on all {n_rows:,} applicants)",
        "",
        f"Scaling: {s.base_score:g} points at good:bad odds of {s.base_odds:g}:1, "
        f"{s.pdo:g} points to double the odds (factor {s.factor:.4f}, offset {s.offset:.4f}).",
        f"Score = sum of the points of the applicant's bins. Cost-optimal cutoff: reject when "
        f"PD >= {threshold:.4f}, i.e. score <= {s.score_from_pd(threshold):.1f}.",
        "Points are rounded here; the service uses the unrounded values (`scorecard.csv`).",
        "`Bin` uses the Statlog labels shipped by OpenML; `Meaning` is what the codes mean",
        "according to Grömping (2019), whose corrected code table fixes 8 mislabelled fields.",
        "",
        "| Feature | Bin | n | Bad rate | WoE | Points | Meaning (Grömping 2019) |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for _, r in table.iterrows():
        lines.append(f"| {r.feature} | {_md(r.bin)} | {r.n} | {r.bad_rate:.1%} | {r.woe:+.3f} | "
                     f"{r.points:.0f} | {_md(r.meaning_groemping_2019)} |")
    (reports / "scorecard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _md(text: str) -> str:
    """Escape pipes (used to join merged levels) inside a markdown table cell."""
    return str(text).replace("|", "\\|")


def _write_iv(iv: pd.DataFrame, reports: Path, n_rows: int, n_folds: int) -> None:
    iv.round(4).to_csv(reports / "iv_table.csv", index=False)
    lines = [f"# Information Value (all {n_rows:,} applicants, descriptive)", "",
             "Inside cross-validation the binning and the IV >= 0.02 filter are refitted on each "
             f"training fold; `CV selection rate` is the share of the {n_folds} folds in which "
             "the scorecard kept the feature.", "",
             "| Feature | IV | Strength | Bins | Excluded by policy | CV selection rate |",
             "|---|---:|---|---:|---|---:|"]
    for _, r in iv.iterrows():
        lines.append(f"| {r.feature} | {r.iv:.3f} | {r.strength} | {r.n_bins} | "
                     f"{'yes' if r.excluded_by_policy else 'no'} | {r.cv_selection_rate:.0%} |")
    (reports / "iv_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _round(obj: Any, nd: int = 6) -> Any:
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {str(k): _round(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round(v, nd) for v in obj]
    if isinstance(obj, np.generic):
        return _round(obj.item(), nd)
    return obj


def run_experiment(X: pd.DataFrame, y: pd.Series, reports_dir: str | Path = "reports",
                   model_path: str | Path = "models/scorecard.joblib", n_splits: int = 5,
                   n_repeats: int = 5, seed: int = 42, hgb_grid: dict[str, list] | None = None,
                   logreg_grid: dict[str, list] | None = None, n_boot: int = 2000,
                   check_codebook: bool = True, make_figures: bool = True,
                   log: Callable[[str], None] = print) -> dict[str, Any]:
    """Run every analysis, write reports/ and the served model; return the metrics dict."""
    reports = Path(reports_dir)
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    yv = np.asarray(y).astype("int64")

    log(f"repeated CV: {n_repeats} x {n_splits} folds")
    cv: CVResult = repeated_cv(X, y, build_specs(hgb_grid, logreg_grid), n_splits=n_splits,
                               n_repeats=n_repeats, seed=seed,
                               importance_for=("hgb", SERVED_MODEL), log=log)
    cv_summary = cv.summary()

    log("calibration, cost and fairness analyses")
    calibration = {m: calibration_summary(yv, o) for m, o in cv.oof.items()}
    cost = {m: cost_analysis(yv, o, seed) for m, o in cv.oof.items()}
    threshold = cost[SERVED_MODEL]["pooled_optimal_threshold"]
    fairness = {
        SERVED_MODEL: fairness_section(X, yv, cv.oof[SERVED_MODEL], threshold, n_boot, seed),
        ABLATION_MODEL: fairness_section(
            X, yv, cv.oof[ABLATION_MODEL], cost[ABLATION_MODEL]["pooled_optimal_threshold"],
            n_boot, seed),
    }

    log("IV table and final scorecard")
    full_binner = WoEBinner().fit(X[list(FEATURES)], yv)
    iv = full_binner.iv_table().drop(columns="selected")
    iv["excluded_by_policy"] = iv["feature"].isin(EXCLUDED_FEATURES)
    sel = Counter(f for fold in cv.scorecard_features[SERVED_MODEL] for f in fold)
    n_folds = len(cv.scorecard_features[SERVED_MODEL])
    iv["cv_selection_rate"] = [sel.get(f, 0) / n_folds for f in iv["feature"]]
    _write_iv(iv, reports, len(yv), n_folds)

    artifact = train_scorecard(X, y, n_splits=n_splits, n_repeats=n_repeats, seed=seed,
                               oof=cv.oof[SERVED_MODEL])
    save_artifact(artifact, model_path)
    scorecard = Scorecard(artifact["pipeline"], artifact["scaler"])
    table = scorecard_report(scorecard)
    _write_scorecard(table, reports, scorecard, artifact["threshold_pd"])
    scores = scorecard.score(X)
    pds = scorecard.predict_pd(X)
    rejected = np.flatnonzero(pds >= artifact["threshold_pd"])[:3]
    examples = [{"row": int(i), "pd": float(pds[i]), "score": float(scores[i]),
                 "reason_codes": scorecard.reason_codes(X.iloc[[i]])[0]} for i in rejected]

    hgb_params = Counter(json.dumps(p, sort_keys=True) for p in cv.best_params.get("hgb", []))
    lr_params = Counter(json.dumps(p, sort_keys=True)
                        for p in cv.best_params.get("logreg_onehot", []))
    importance = {name: [{"feature": f, "mean": float(r["mean"]), "std": float(r["std"])}
                         for f, r in df.iterrows()] for name, df in cv.importance.items()}
    metrics: dict[str, Any] = {
        "data": {
            "source": "OpenML data_id=31 (credit-g = Statlog German Credit, Hofmann 1994)",
            "n_applicants": int(len(yv)), "bad_rate": float(yv.mean()),
            "n_raw_features": len(FEATURES), "model_features": list(MODEL_FEATURES),
            "excluded_features": list(EXCLUDED_FEATURES),
            "codebook_check_max_gap_pp": codebook_check(X, y) if check_codebook else None,
        },
        "setup": {
            "cv": f"RepeatedStratifiedKFold(n_splits={n_splits}, n_repeats={n_repeats}, "
                  f"random_state={seed})",
            "cost_matrix": {"approve_bad": COST_APPROVE_BAD, "reject_good": COST_REJECT_GOOD},
            "scorecard_scaling": {"base_score": scorecard.scaler.base_score,
                                  "base_odds": scorecard.scaler.base_odds,
                                  "pdo": scorecard.scaler.pdo},
            "hgb_grid": hgb_grid or HGB_GRID, "logreg_grid": logreg_grid or LOGREG_GRID,
        },
        "cv_metrics": cv_summary,
        "calibration_pooled_oof": calibration,
        "cost": {
            "baselines": {"approve_all": float(COST_APPROVE_BAD * yv.mean()),
                          "reject_all": float(COST_REJECT_GOOD * (1 - yv.mean()))},
            "models": cost,
        },
        "fairness": {"threshold_pd": threshold, "models": fairness},
        "tuning": {"hgb_best_params_counts": dict(hgb_params.most_common()),
                   "logreg_best_params_counts": dict(lr_params.most_common())},
        "permutation_importance_roc_auc": importance,
        "final_model": {
            "model": SERVED_MODEL, "features": scorecard.features,
            "n_bins": int(len(table)), "threshold_pd": artifact["threshold_pd"],
            "cutoff_score": artifact["cutoff_score"],
            "oof_cost_per_applicant": artifact["oof_cost_per_applicant"],
            "oof_approval_rate": artifact["oof_approval_rate"],
            "in_sample_score_range": [float(scores.min()), float(scores.max())],
            "in_sample_score_median": float(np.median(scores)),
            "example_rejections": examples,
        },
    }
    metrics = _round(metrics)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    if make_figures:
        from credit_risk import figures

        log("figures")
        figures.make_all(X, yv, cv, metrics, reports / "figures")
    return metrics
