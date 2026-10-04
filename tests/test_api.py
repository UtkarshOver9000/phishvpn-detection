"""API tests against the shipped model artifact in src/phishurl/artifacts."""

from fastapi.testclient import TestClient

from phishurl.api.app import app
from phishurl.scorer import risk_tier

client = TestClient(app)


def test_health():
    assert client.get("/v1/health").json() == {"status": "ok"}


def test_dashboard_is_served():
    res = client.get("/")
    assert res.status_code == 200 and "Phishing URL Detection" in res.text


def test_stats_come_from_the_model_card():
    data = client.get("/v1/stats").json()
    assert data["model"] == "gradient_boosting"
    assert 0 < data["threshold"] < 1
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "log_loss"):
        assert 0 <= data["test_metrics"][key] <= 1 or key == "log_loss"
    assert data["live_evaluation"]["tranco_list_id"]
    assert "false_alarms_per_10k_legit_sites" in data["business"]


def test_score_response_shape():
    res = client.post("/v1/score", json={"url": "https://secure-account-verify-paypa1.com/login"})
    assert res.status_code == 200
    data = res.json()
    assert data["registrable_domain"] == "secure-account-verify-paypa1.com"
    assert 0 <= data["phishing_probability"] <= 1
    assert data["flagged"] == (data["phishing_probability"] >= data["threshold"])
    assert data["risk_tier"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert isinstance(data["reasons"], list)


def test_path_does_not_change_the_score():
    a = client.post("/v1/score", json={"url": "https://example.org"}).json()
    b = client.post("/v1/score", json={"url": "http://www.example.org/login/verify?x=1"}).json()
    assert a["phishing_probability"] == b["phishing_probability"]


def test_rejects_input_without_a_domain():
    assert client.post("/v1/score", json={"url": "://"}).status_code == 422
    assert client.post("/v1/score", json={"url": ""}).status_code == 422


def test_risk_tier_is_anchored_on_threshold():
    assert risk_tier(0.95, 0.6) == "CRITICAL"
    assert risk_tier(0.65, 0.6) == "HIGH"
    assert risk_tier(0.35, 0.6) == "MEDIUM"
    assert risk_tier(0.1, 0.6) == "LOW"
