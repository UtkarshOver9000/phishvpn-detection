"""
FastAPI wrapper around the phishing-detection model for interactive use.

Trains a small in-memory model from synthetic data at import time (same
pattern as the sibling ittravel project's AIAnomalyEngine) rather than
loading a persisted artifact, so the API has no external model file to
manage or go stale -- it always reflects the current synthetic_data.py /
model.py code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..data_schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from ..model import evaluate_model, train_model
from ..synthetic_data import generate_synthetic

TRAIN_ROWS = 8000
TRAIN_SEED = 7


class VpnSession(BaseModel):
    country: str = "US"
    region: str = "na"
    vpn_provider: str = "provider_a"
    protocol: str = "openvpn"
    device_type: str = "desktop"
    auth_method: str = "password"
    mfa_used: str = "yes"
    login_failures_24h: int = 0
    unique_ips_24h: int = 1
    session_duration_s: float = 600.0
    domain_similarity_score: float = 0.2
    suspicious_url_count: int = 0
    cert_age_days: float = 200.0
    new_account_days: int = 30
    account_age_days: int = 400
    hour_of_day: int = 12
    day_of_week: int = 2


class ScoreResult(BaseModel):
    phishing_probability: float
    is_phishing_prediction: bool
    risk_tier: str
    timestamp: str


def risk_tier(prob: float) -> str:
    if prob >= 0.8:
        return "CRITICAL"
    if prob >= 0.6:
        return "HIGH"
    if prob >= 0.35:
        return "MEDIUM"
    return "LOW"


def bootstrap_model():
    df = generate_synthetic(TRAIN_ROWS, seed=TRAIN_SEED)
    result = train_model(df, test_size=0.2, seed=TRAIN_SEED)
    metrics = evaluate_model(result.model, result.x_test, result.y_test)
    return result.model, metrics


_model, _training_metrics = bootstrap_model()

app = FastAPI(
    title="PhishVPN Detection API",
    description=(
        "Phishing-risk scoring for suspicious VPN sessions. The model is trained "
        "in-memory from synthetic data on startup -- see the repo README for real, "
        "measured precision/recall/ROC-AUC and an explanation of two bugs that were "
        "fixed to get there."
    ),
    version="1.0.0",
    contact={"name": "phishvpn-detection", "url": "https://github.com/UtkarshOver9000/phishvpn-detection"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    index_file = DASHBOARD_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")
    return HTMLResponse("<h1>PhishVPN Detection API</h1><p>Visit <a href='/docs'>/docs</a></p>")


@app.post("/v1/score", response_model=ScoreResult, summary="Score a VPN session", tags=["Scoring"])
async def score_session(session: VpnSession):
    """Return a phishing-risk probability and tier for a single VPN session."""
    row = pd.DataFrame([session.model_dump()])[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]
    prob = float(_model.predict_proba(row)[0, 1])
    return ScoreResult(
        phishing_probability=round(prob, 4),
        is_phishing_prediction=prob >= 0.5,
        risk_tier=risk_tier(prob),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/v1/stats", summary="Model training metrics", tags=["Telemetry"])
async def get_stats():
    """Real metrics from the in-memory model's own held-out split, computed at startup."""
    report = _training_metrics["classification_report"]
    return {
        "engine_status": "ONLINE",
        "model": "LogisticRegression (scikit-learn)",
        "trained_on_rows": TRAIN_ROWS,
        "roc_auc": round(_training_metrics["roc_auc"], 4),
        "phishing_class_precision": round(report["1"]["precision"], 4),
        "phishing_class_recall": round(report["1"]["recall"], 4),
        "accuracy": round(report["accuracy"], 4),
    }


@app.get("/v1/health", include_in_schema=False)
async def health_check():
    return {"status": "ok"}
