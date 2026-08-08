"""
Unit tests for the FastAPI wrapper around the phishing-detection model.
"""

from fastapi.testclient import TestClient

from phishvpn.api.app import app, risk_tier

client = TestClient(app)

BENIGN_SESSION = {
    "country": "US", "region": "na", "vpn_provider": "provider_a", "protocol": "wireguard",
    "device_type": "desktop", "auth_method": "sso", "mfa_used": "yes",
    "login_failures_24h": 0, "unique_ips_24h": 1, "session_duration_s": 900,
    "domain_similarity_score": 0.05, "suspicious_url_count": 0, "cert_age_days": 400,
    "new_account_days": 45, "account_age_days": 1200, "hour_of_day": 14, "day_of_week": 2,
}

SUSPICIOUS_SESSION = {
    "country": "RU", "region": "eu", "vpn_provider": "unknown", "protocol": "pptp",
    "device_type": "mobile", "auth_method": "password", "mfa_used": "no",
    "login_failures_24h": 5, "unique_ips_24h": 7, "session_duration_s": 120,
    "domain_similarity_score": 0.92, "suspicious_url_count": 4, "cert_age_days": 4,
    "new_account_days": 1, "account_age_days": 3, "hour_of_day": 3, "day_of_week": 6,
}


def test_risk_tier_boundaries():
    assert risk_tier(0.0) == "LOW"
    assert risk_tier(0.34) == "LOW"
    assert risk_tier(0.35) == "MEDIUM"
    assert risk_tier(0.6) == "HIGH"
    assert risk_tier(0.8) == "CRITICAL"
    assert risk_tier(1.0) == "CRITICAL"


def test_health_check():
    res = client.get("/v1/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_stats_reports_real_training_metrics():
    res = client.get("/v1/stats")
    assert res.status_code == 200
    data = res.json()
    assert data["engine_status"] == "ONLINE"
    assert 0.0 <= data["roc_auc"] <= 1.0
    assert 0.0 <= data["phishing_class_recall"] <= 1.0
    assert data["trained_on_rows"] > 0


def test_score_benign_session_is_low_risk():
    res = client.post("/v1/score", json=BENIGN_SESSION)
    assert res.status_code == 200
    data = res.json()
    assert data["phishing_probability"] < 0.35
    assert data["risk_tier"] == "LOW"
    assert data["is_phishing_prediction"] is False


def test_score_suspicious_session_is_high_risk():
    res = client.post("/v1/score", json=SUSPICIOUS_SESSION)
    assert res.status_code == 200
    data = res.json()
    assert data["phishing_probability"] > 0.8
    assert data["risk_tier"] == "CRITICAL"
    assert data["is_phishing_prediction"] is True


def test_score_uses_field_defaults_when_omitted():
    res = client.post("/v1/score", json={})
    assert res.status_code == 200
    data = res.json()
    assert 0.0 <= data["phishing_probability"] <= 1.0


def test_dashboard_root_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "PhishVPN" in res.text
