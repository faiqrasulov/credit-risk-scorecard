import pytest

from credit_risk.data import make_synthetic_credit


@pytest.fixture(scope="session")
def synthetic():
    """Synthetic applicants with the German Credit schema (no network, known signal)."""
    return make_synthetic_credit(n=600, seed=7)
