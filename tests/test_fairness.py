import numpy as np
import pytest

from credit_risk.fairness import (
    age_group,
    fairness_audit,
    group_rates,
    sex_corrected,
    sex_statlog,
)


def test_group_rates_known_answer():
    y = np.array([0, 0, 1, 1, 0, 0])
    approve = np.array([1, 0, 1, 0, 1, 1])
    g = np.array(["a", "a", "a", "b", "b", "b"], dtype=object)
    t = group_rates(y, approve, g)
    assert t.loc["a", "approval_rate"] == pytest.approx(2 / 3)
    assert t.loc["a", "tpr"] == pytest.approx(0.5) and t.loc["a", "fpr"] == pytest.approx(1.0)
    assert t.loc["b", "tpr"] == pytest.approx(1.0) and t.loc["b", "fpr"] == pytest.approx(0.0)
    audit = fairness_audit(y, approve, g, unprivileged="a", privileged="b", n_boot=200)
    assert audit["disparate_impact"] == pytest.approx(1.0)
    assert audit["equal_opportunity_difference"] == pytest.approx(-0.5)
    lo, hi = audit["groups_ci95"]["a"]["approval_rate"]
    assert 0 <= lo <= hi <= 1


def test_attribute_derivations():
    assert list(age_group([19, 24.9, 25, 60])) == ["<25", "<25", ">=25", ">=25"]
    ps = ["male div/sep", "female div/dep/mar", "male single", "male mar/wid"]
    assert list(sex_statlog(ps)) == ["male", "female", "male", "male"]
    # Grömping (2019): A92 mixes non-single women with single men; A94 is single women.
    assert list(sex_corrected(ps)) == ["male", "mixed", "male", "female"]
