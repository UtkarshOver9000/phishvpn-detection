"""
Load the shipped model and score URLs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib

from .explain import reasons
from .features import domain_features, featurize, split_domain

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


@dataclass(frozen=True)
class Score:
    url: str
    registrable_domain: str
    phishing_probability: float
    flagged: bool
    risk_tier: str
    reasons: list[str]
    features: dict


def risk_tier(prob: float, threshold: float) -> str:
    """Tiers are anchored on the operating threshold chosen for a 1% false-positive rate."""
    if prob >= max(0.9, threshold):
        return "CRITICAL"
    if prob >= threshold:
        return "HIGH"
    if prob >= threshold / 2:
        return "MEDIUM"
    return "LOW"


class Scorer:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR):
        self.model = joblib.load(artifact_dir / "model.joblib")
        self.card = json.loads((artifact_dir / "model_card.json").read_text())
        self.threshold = float(self.card["threshold"])

    def score(self, url: str) -> Score:
        feats = domain_features(url)
        prob = float(self.model.predict_proba(featurize([url]))[0, 1])
        return Score(
            url=url,
            registrable_domain=split_domain(url).registrable,
            phishing_probability=round(prob, 4),
            flagged=prob >= self.threshold,
            risk_tier=risk_tier(prob, self.threshold),
            reasons=reasons(feats, self.card["legit_reference_ranges"]),
            features=feats,
        )


@lru_cache(maxsize=1)
def default_scorer() -> Scorer:
    return Scorer()
