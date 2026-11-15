import pytest
from fastapi.testclient import TestClient

from credit_risk.api import create_app
from credit_risk.api_schema import EXAMPLE_APPLICANT
from credit_risk.train import load_artifact, save_artifact, train_scorecard


@pytest.fixture(scope="module")
def artifact(synthetic):
    X, y = synthetic
    return train_scorecard(X, y, n_splits=3, n_repeats=1, seed=0)


@pytest.fixture(scope="module")
def client(artifact):
    with TestClient(create_app(artifact=artifact)) as c:
        yield c


def test_health(client, artifact):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model"] == "woe_scorecard"
    assert body["threshold_pd"] == pytest.approx(artifact["threshold_pd"])


def test_score_returns_consistent_decision_and_reasons(client, artifact):
    r = client.post("/score", json=EXAMPLE_APPLICANT)
    assert r.status_code == 200
    body = r.json()
    pd_ = body["probability_of_default"]
    assert 0 < pd_ < 1
    assert body["decision"] == ("reject" if pd_ >= artifact["threshold_pd"] else "approve")
    assert body["score"] == pytest.approx(float(artifact["scaler"].score_from_pd(pd_)), abs=0.02)
    reasons = body["reason_codes"]
    assert len(reasons) <= 3
    lost = [rc["points_lost"] for rc in reasons]
    assert lost == sorted(lost, reverse=True) and all(v > 0 for v in lost)


def test_riskier_applicant_scores_lower(client):
    safe = dict(EXAMPLE_APPLICANT, checking_status="no checking", duration=6)
    risky = dict(EXAMPLE_APPLICANT, checking_status="<0", duration=60, credit_amount=15000)
    s_safe = client.post("/score", json=safe).json()["score"]
    s_risky = client.post("/score", json=risky).json()["score"]
    assert s_risky < s_safe


@pytest.mark.parametrize("patch", [
    {"checking_status": "lots of money"},  # unknown category
    {"duration": 0},  # out of range
    {"installment_commitment": 7},  # not a valid code
    {"favourite_colour": "blue"},  # unexpected field
])
def test_invalid_requests_are_rejected(client, patch):
    assert client.post("/score", json=dict(EXAMPLE_APPLICANT, **patch)).status_code == 422


def test_missing_field_is_rejected(client):
    body = {k: v for k, v in EXAMPLE_APPLICANT.items() if k != "age"}
    assert client.post("/score", json=body).status_code == 422


def test_artifact_roundtrip_and_missing_model(artifact, tmp_path):
    path = save_artifact(artifact, tmp_path / "m.joblib")
    assert load_artifact(path)["threshold_pd"] == artifact["threshold_pd"]
    with TestClient(create_app(model_path=path)) as c:
        assert c.get("/health").status_code == 200
    with TestClient(create_app(model_path=tmp_path / "missing.joblib")) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/score", json=EXAMPLE_APPLICANT).status_code == 503
