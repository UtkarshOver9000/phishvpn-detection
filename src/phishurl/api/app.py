"""
FastAPI service and demo page for the phishing-domain model.

The model is trained offline on real data (``python -m phishurl.train``) and
loaded from ``src/phishurl/artifacts``. Nothing is trained at startup.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .. import __version__
from ..scorer import default_scorer

app = FastAPI(
    title="Phishing URL Detection API",
    description=(
        "Scores how likely a URL's registrable domain is to be a phishing domain. Trained on the PhiUSIIL "
        "dataset plus Tranco top sites, and checked against the live OpenPhish feed. Metrics: /v1/stats."
    ),
    version=__version__,
    license_info={"name": "MIT"},
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

DASHBOARD = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"


class ScoreRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048, examples=["https://secure-login-paypa1.com/verify"])


class ScoreResponse(BaseModel):
    url: str
    registrable_domain: str
    phishing_probability: float
    flagged: bool
    risk_tier: str
    threshold: float
    reasons: list[str]
    features: dict
    timestamp: str


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    if DASHBOARD.exists():
        return FileResponse(str(DASHBOARD), media_type="text/html")
    return HTMLResponse("<h1>Phishing URL Detection API</h1><p>See <a href='/docs'>/docs</a></p>")


@app.post("/v1/score", response_model=ScoreResponse, tags=["Scoring"])
async def score(req: ScoreRequest):
    """Score one URL. Only its registrable domain is used, so the path never changes the result."""
    scorer = default_scorer()
    result = scorer.score(req.url)
    if not result.registrable_domain:
        raise HTTPException(status_code=422, detail="Could not find a domain in that URL")
    return ScoreResponse(
        **result.__dict__,
        threshold=scorer.threshold,
        timestamp=datetime.now(UTC).isoformat(),
    )


@app.get("/v1/stats", tags=["Model"])
async def stats():
    """Held-out and live-data metrics of the deployed model, read from its model card."""
    card = default_scorer().card
    return {
        "model": card["model"],
        "threshold": card["threshold"],
        "test_metrics": card["test_metrics"],
        "live_evaluation": card["live_evaluation"],
        "business": card["business"],
        "scikit_learn": card["scikit_learn"],
    }


@app.get("/v1/health", include_in_schema=False)
async def health():
    return {"status": "ok"}
