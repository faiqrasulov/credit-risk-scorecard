import numpy as np
import pandas as pd
import pytest

from credit_risk.data import codebook_check, make_synthetic_credit
from credit_risk.schema import (
    CATEGORICAL_LEVELS,
    CODED_NUMERIC_LEVELS,
    CORRECTED_MEANING,
    FEATURES,
    GROEMPING_TABLE1,
    corrected_meaning,
)


def test_code_tables_are_aligned():
    assert len(FEATURES) == 20
    for f, meanings in CORRECTED_MEANING.items():
        levels = CATEGORICAL_LEVELS.get(f) or CODED_NUMERIC_LEVELS[f]
        assert len(meanings) == len(levels), f


@pytest.mark.parametrize("feature", sorted(GROEMPING_TABLE1))
def test_transcribed_table1_is_consistent(feature):
    """Percentages must sum to 100 and imply whole numbers of the 300 bads / 700 goods."""
    bad, good = (np.array(v) for v in GROEMPING_TABLE1[feature])
    assert bad.sum() == pytest.approx(100, abs=0.05) and good.sum() == pytest.approx(100, abs=0.05)
    np.testing.assert_allclose(bad * 3, np.round(bad * 3), atol=0.02)
    np.testing.assert_allclose(good * 7, np.round(good * 7), atol=0.04)


def test_codebook_check_on_data_built_from_table1():
    rng = np.random.default_rng(0)
    cols = {}
    for f, (bad, good) in GROEMPING_TABLE1.items():
        levels = CATEGORICAL_LEVELS.get(f) or CODED_NUMERIC_LEVELS[f]
        b = np.repeat(levels, np.round(np.array(bad) * 3).astype(int))
        g = np.repeat(levels, np.round(np.array(good) * 7).astype(int))
        cols[f] = np.concatenate([rng.permutation(b), rng.permutation(g)])
    y = pd.Series([1] * 300 + [0] * 700)
    gaps = codebook_check(pd.DataFrame(cols), y)
    assert max(gaps.values()) < 0.01
    X_syn, y_syn = make_synthetic_credit(400, seed=1)
    assert max(codebook_check(X_syn, y_syn).values()) > 1  # random data does not match


def test_corrected_meaning_lookup():
    assert corrected_meaning("checking_status", "no checking") == (
        ">= 200 DM or salary for at least 1 year")
    assert corrected_meaning("credit_history", "critical/other existing credit") == (
        "all credits at this bank paid back duly")
    assert corrected_meaning("installment_commitment", 4.0) == "< 20%"
    assert corrected_meaning("duration", 12) is None
