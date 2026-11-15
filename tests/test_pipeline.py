import json

import numpy as np
import pandas as pd
import pytest

from credit_risk.cli import main
from credit_risk.data import MODEL_FEATURES, clean_frame, make_synthetic_credit
from credit_risk.experiment import run_experiment
from credit_risk.models import make_hgb

TINY_HGB = {"hgb__max_iter": [30], "hgb__max_leaf_nodes": [7], "hgb__min_samples_leaf": [20]}


def test_end_to_end_on_synthetic_data(tmp_path):
    X, y = make_synthetic_credit(n=500, seed=3)
    m = run_experiment(X, y, reports_dir=tmp_path / "reports",
                       model_path=tmp_path / "models" / "m.joblib", n_splits=3, n_repeats=2,
                       seed=0, hgb_grid=TINY_HGB, logreg_grid={"logreg__C": [1.0]}, n_boot=50,
                       check_codebook=False, log=lambda s: None)
    reports = tmp_path / "reports"
    for f in ("metrics.json", "scorecard.csv", "scorecard.md", "iv_table.csv", "iv_table.md"):
        assert (reports / f).exists(), f
    for f in ("model_comparison", "calibration", "cost_vs_threshold", "fairness"):
        assert (reports / "figures" / f"{f}.png").stat().st_size > 10_000
    assert (tmp_path / "models" / "m.joblib").exists()
    saved = json.loads((reports / "metrics.json").read_text())
    assert saved["final_model"]["threshold_pd"] == m["final_model"]["threshold_pd"]
    for model in ("woe_scorecard", "logreg_onehot", "hgb", "hgb_isotonic", "hgb_sigmoid"):
        assert saved["cv_metrics"][model]["roc_auc"]["mean"] > 0.6, model
    cost = saved["cost"]["models"]["woe_scorecard"]
    assert cost["optimal_in_sample"]["cost"]["mean"] <= cost["threshold_0.5"]["cost"]["mean"]
    assert set(saved["fairness"]["models"]["woe_scorecard"]) == {
        "age", "sex_statlog_codes", "sex_corrected_codes"}
    table = pd.read_csv(reports / "scorecard.csv")
    assert {"feature", "bin", "points"} <= set(table.columns)


def test_hgb_pipeline_handles_unseen_category():
    X, y = make_synthetic_credit(n=300, seed=4)
    feats = list(MODEL_FEATURES)
    model = make_hgb(feats).set_params(hgb__max_iter=20).fit(X[feats], y)
    row = X.iloc[[0]].copy()
    row["purpose"] = "vacation"  # valid Statlog code never seen in training
    p = model.predict_proba(clean_frame(row)[feats])[:, 1]
    assert np.isfinite(p).all()


def test_cli_rejects_unknown_command(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["nonsense"])
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
