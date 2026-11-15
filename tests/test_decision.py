import numpy as np
import pytest

from credit_risk.decision import (
    approval_rate,
    bayes_threshold,
    cost_curve,
    cross_validated_cost,
    expected_cost,
    optimal_threshold,
)


def brute_force_cost(y, p, t, c_fn=5.0, c_fp=1.0):
    total = 0.0
    for yi, pi in zip(y, p, strict=True):
        reject = pi >= t
        if yi == 1 and not reject:
            total += c_fn
        elif yi == 0 and reject:
            total += c_fp
    return total / len(y)


def test_known_answer():
    # Rejecting PD >= 0.1 rejects both bads and two goods: cost (0 * 5 + 2 * 1) / 5 = 0.4.
    p = np.array([0.05, 0.1, 0.2, 0.3, 0.6])
    y = np.array([0, 1, 0, 0, 1])
    t, c = optimal_threshold(y, p)
    assert t == pytest.approx(0.075) and c == pytest.approx(0.4)
    assert expected_cost(y, p, 0.5) == pytest.approx(1.0)  # one bad approved
    assert expected_cost(y, p, 1.0) == pytest.approx(2.0)  # approve everyone
    assert expected_cost(y, p, 0.0) == pytest.approx(0.6)  # reject everyone
    assert approval_rate(p, 0.075) == pytest.approx(0.2)


def test_optimum_matches_brute_force_search():
    rng = np.random.default_rng(0)
    p = np.round(rng.random(300), 2)  # ties on purpose
    y = (rng.random(300) < p).astype(int)
    t, c = optimal_threshold(y, p)
    candidates = np.concatenate([np.unique(p), [1.01]])
    brute = min(brute_force_cost(y, p, ti) for ti in candidates)
    assert c == pytest.approx(brute)
    assert brute_force_cost(y, p, t) == pytest.approx(c)
    grid = np.linspace(0, 1, 101)
    np.testing.assert_allclose(cost_curve(y, p, grid),
                               [brute_force_cost(y, p, ti) for ti in grid])


def test_bayes_threshold_and_calibrated_simulation():
    assert bayes_threshold() == pytest.approx(1 / 6)
    assert bayes_threshold(cost_approve_bad=1, cost_reject_good=1) == pytest.approx(0.5)
    rng = np.random.default_rng(1)
    p = rng.random(200_000)
    y = (rng.random(200_000) < p).astype(int)  # perfectly calibrated PDs
    t, _ = optimal_threshold(y, p)
    assert t == pytest.approx(1 / 6, abs=0.02)


def test_cross_validated_cost_is_bounded():
    rng = np.random.default_rng(2)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    c = cross_validated_cost(y, p, n_splits=5, seed=0)
    _, c_in = optimal_threshold(y, p)
    assert c_in - 0.05 < c < min(5 * y.mean(), 1 - y.mean())
