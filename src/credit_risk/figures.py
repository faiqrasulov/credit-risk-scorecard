"""Report figures (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from credit_risk.decision import bayes_threshold, cost_curve  # noqa: E402
from credit_risk.evaluation import CVResult  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
STYLE = {
    "woe_scorecard": dict(color=BLUE, ls="-", label="WoE scorecard"),
    "logreg_onehot": dict(color=ORANGE, ls="-", label="Logistic regression (one-hot)"),
    "hgb": dict(color=AQUA, ls="-", label="Gradient boosting (HGB)"),
    "hgb_sigmoid": dict(color=AQUA, ls="--", label="HGB + sigmoid calibration"),
    "hgb_isotonic": dict(color=AQUA, ls="--", label="HGB + isotonic calibration"),
}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2,
    })


def _best_calibrated(metrics: dict[str, Any]) -> str:
    cal = metrics["calibration_pooled_oof"]
    return min(("hgb_sigmoid", "hgb_isotonic"), key=lambda m: cal[m]["brier"])


def model_comparison(y: np.ndarray, cv: CVResult, metrics: dict[str, Any], path: Path) -> None:
    """ROC curves (first CV repeat) and fold-level metric spread for the three model families."""
    models = ["woe_scorecard", "logreg_onehot", "hgb"]
    summ = metrics["cv_metrics"]
    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.05, 1], hspace=0.95, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0])
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=":")
    for m in models:
        fpr, tpr, _ = roc_curve(y, cv.oof[m][0])
        s = summ[m]["roc_auc"]
        ax.plot(fpr, tpr, color=STYLE[m]["color"], ls=STYLE[m]["ls"],
                label=f"{STYLE[m]['label']}: AUC {s['mean']:.3f} ± {s['std']:.3f}")
    ax.set(xlabel="False-positive rate (goods flagged as bad)",
           ylabel="True-positive rate (bads caught)", xlim=(0, 1), ylim=(0, 1.01),
           title="ROC, out-of-fold (repeat 1 of 5)")
    ax.legend(loc="lower right", fontsize=9)
    specs = [("roc_auc", "ROC-AUC (higher is better)"), ("pr_auc", "PR-AUC (higher is better)"),
             ("ks", "KS statistic (higher is better)"), ("brier", "Brier score (lower is better)")]
    for i, (key, title) in enumerate(specs):
        a = fig.add_subplot(gs[i, 1])
        for j, m in enumerate(models):
            s = summ[m][key]
            a.errorbar(s["mean"], j, xerr=s["std"], fmt="o", ms=8, color=STYLE[m]["color"],
                       ecolor=STYLE[m]["color"], elinewidth=2, capsize=0)
            a.annotate(f"{s['mean']:.3f}", (s["mean"] + s["std"], j), xytext=(5, 0),
                       textcoords="offset points", ha="left", va="center", fontsize=8,
                       color=INK_2)
        lo = min(summ[m][key]["mean"] - summ[m][key]["std"] for m in models)
        hi = max(summ[m][key]["mean"] + summ[m][key]["std"] for m in models)
        a.set_xlim(lo - 0.05 * (hi - lo), hi + 0.22 * (hi - lo))
        a.set_yticks(range(len(models)), ["Scorecard", "LogReg", "HGB"], fontsize=8)
        a.set_ylim(-0.6, len(models) - 0.1)
        a.invert_yaxis()
        a.set_title(title, fontsize=9.5, loc="left")
        a.grid(axis="y", visible=False)
    fig.text(0.53, 0.005, "Right: mean ± 1 SD over 25 folds (5 x 5 repeated stratified CV)",
             fontsize=8.5, color=INK_2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def calibration(y: np.ndarray, cv: CVResult, metrics: dict[str, Any], path: Path,
                n_bins: int = 10) -> None:
    """Reliability diagram on pooled out-of-fold PDs (equal-count bins)."""
    cal = metrics["calibration_pooled_oof"]
    models = ["woe_scorecard", "logreg_onehot", "hgb", _best_calibrated(metrics)]
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.plot([0, 1], [0, 1], color=INK_2, lw=1, ls=":", label="Perfect calibration")
    yy = np.tile(y, cv.oof["woe_scorecard"].shape[0])
    for m in models:
        p = cv.oof[m].ravel()
        order = np.argsort(p, kind="stable")
        chunks = np.array_split(order, n_bins)
        xs = [p[c].mean() for c in chunks]
        ys = [yy[c].mean() for c in chunks]
        c = cal[m]
        ax.plot(xs, ys, marker="o", ms=6, color=STYLE[m]["color"], ls=STYLE[m]["ls"],
                label=f"{STYLE[m]['label']} (Brier {c['brier']:.3f}, ECE {c['ece']:.3f}, "
                      f"slope {c['calibration_slope']:.2f})")
    ax.set(xlabel="Mean predicted PD (decile of predictions)", ylabel="Observed bad rate",
           xlim=(0, 0.9), ylim=(0, 0.9), title="Calibration, pooled out-of-fold predictions")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cost_vs_threshold(y: np.ndarray, cv: CVResult, metrics: dict[str, Any], path: Path) -> None:
    """Expected cost per applicant and approval rate as the PD cutoff varies."""
    models = ["woe_scorecard", "logreg_onehot", _best_calibrated(metrics)]
    grid = np.linspace(0.0, 1.0, 401)
    base = metrics["cost"]["baselines"]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.2, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1], "hspace": 0.12})
    for m in models:
        curves = np.array([cost_curve(y, p, grid) for p in cv.oof[m]])
        appr = np.array([[np.mean(p < t) for t in grid] for p in cv.oof[m]])
        ax.plot(grid, curves.mean(axis=0), color=STYLE[m]["color"], ls=STYLE[m]["ls"],
                label=STYLE[m]["label"])
        ax2.plot(grid, appr.mean(axis=0), color=STYLE[m]["color"], ls=STYLE[m]["ls"])
    sc = metrics["cost"]["models"]["woe_scorecard"]
    t_opt = sc["pooled_optimal_threshold"]
    for level, name, x, ha in ((base["reject_all"], "reject everyone", 0.99, "right"),
                               (base["approve_all"], "approve everyone", 0.01, "left")):
        ax.axhline(level, color=INK_2, lw=1, ls=":")
        ax.text(x, level - 0.045, f"{name}: {level:.2f}", ha=ha, fontsize=8.5, color=INK_2)
    for t, name in ((bayes_threshold(), "Bayes 1/6"), (0.5, "0.5")):
        for a in (ax, ax2):
            a.axvline(t, color=INK_2, lw=1, ls="--")
        ax.text(t + 0.01, 1.55, name, fontsize=8.5, color=INK_2)
    c_opt = sc["pooled_optimal_cost"]
    ax.plot([t_opt], [c_opt], "o", ms=9, color=BLUE, mec=SURFACE, mew=2)
    ax.annotate(f"scorecard optimum: PD cutoff {t_opt:.3f}, cost {c_opt:.3f}",
                (t_opt, c_opt), xytext=(0.25, 0.44), textcoords="data", fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=INK_2, lw=1))
    ax.set(ylabel="Expected cost per applicant", ylim=(0.4, 1.6),
           title="Cost of the accept/reject rule (approve bad = 5, reject good = 1)")
    ax.legend(loc="upper center", fontsize=8.5, bbox_to_anchor=(0.5, 0.88))
    ax2.set(xlabel="PD cutoff (reject if PD >= cutoff)", ylabel="Approval rate",
            ylim=(0, 1.02), xlim=(0, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fairness(metrics: dict[str, Any], path: Path) -> None:
    """Approval rate, TPR and FPR by group at the deployed cutoff, with bootstrap 95% CIs."""
    audit = metrics["fairness"]["models"]["woe_scorecard"]
    panels = [("age", "Age", ["<25", ">=25"]),
              ("sex_statlog_codes", "Sex, Statlog code table", ["female", "male"]),
              ("sex_corrected_codes", "Sex, corrected code table", ["female", "male", "mixed"])]
    bars = [("approval_rate", "Approval rate", BLUE), ("tpr", "TPR = P(approve | good)", ORANGE),
            ("fpr", "FPR = P(approve | bad)", AQUA)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True,
                             gridspec_kw={"width_ratios": [2, 2, 3]})
    width = 0.26
    for ax, (key, title, groups) in zip(axes, panels, strict=True):
        a = audit[key]
        x = np.arange(len(groups))
        for k, (metric, label, color) in enumerate(bars):
            vals = np.array([a["groups"][g][metric] for g in groups])
            lo = np.array([a["groups_ci95"][g][metric][0] for g in groups])
            hi = np.array([a["groups_ci95"][g][metric][1] for g in groups])
            ax.bar(x + (k - 1) * width, vals, width * 0.92, color=color, label=label,
                   edgecolor=SURFACE, linewidth=1.5)
            ax.errorbar(x + (k - 1) * width, vals, yerr=[vals - lo, hi - vals], fmt="none",
                        ecolor=INK, elinewidth=1, capsize=2)
        ax.set_xticks(x, [f"{g}\n(n={int(a['groups'][g]['n'])})" for g in groups])
        di, eod = a["disparate_impact"], a["equal_opportunity_difference"]
        ax.set_title(f"{title}\nDI {di:.2f} [{a['disparate_impact_ci95'][0]:.2f}, "
                     f"{a['disparate_impact_ci95'][1]:.2f}], EOD {eod:+.3f}", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set(ylabel="Rate at the deployed cutoff", ylim=(0, 1.05))
    axes[0].legend(loc="upper left", fontsize=8, bbox_to_anchor=(0, -0.2), ncol=3)
    fig.suptitle("Fairness audit of the WoE scorecard (age and sex not used as features); "
                 "error bars = bootstrap 95% CI", fontsize=11, y=1.04)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_all(X: pd.DataFrame, y: np.ndarray, cv: CVResult, metrics: dict[str, Any],
             out_dir: Path) -> None:
    _style()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_comparison(y, cv, metrics, out_dir / "model_comparison.png")
    calibration(y, cv, metrics, out_dir / "calibration.png")
    cost_vs_threshold(y, cv, metrics, out_dir / "cost_vs_threshold.png")
    fairness(metrics, out_dir / "fairness.png")
