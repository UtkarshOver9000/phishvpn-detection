"""
Synthetic data generator for suspicious VPN phishing detection.

Generates realistic distributions for geo, network, and behavioral features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data_schema import ALL_COLUMNS

COUNTRIES = ["US", "IN", "DE", "BR", "GB", "CN", "JP", "ZA", "AE", "SG", "RU", "NG"]
REGIONS = ["na", "sa", "eu", "apac", "mea"]
VPN_PROVIDERS = ["provider_a", "provider_b", "provider_c", "unknown"]
PROTOCOLS = ["openvpn", "wireguard", "ikev2", "pptp"]
DEVICE_TYPES = ["desktop", "mobile", "server"]
AUTH_METHODS = ["password", "sso", "token"]
MFA_USED = ["yes", "no"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


def generate_synthetic(rows: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    country = rng.choice(COUNTRIES, size=rows, p=[0.2, 0.18, 0.08, 0.08, 0.07, 0.1, 0.06, 0.05, 0.05, 0.05, 0.05, 0.03])
    region = rng.choice(REGIONS, size=rows, p=[0.35, 0.1, 0.2, 0.25, 0.1])
    vpn_provider = rng.choice(VPN_PROVIDERS, size=rows, p=[0.35, 0.25, 0.2, 0.2])
    protocol = rng.choice(PROTOCOLS, size=rows, p=[0.45, 0.3, 0.2, 0.05])
    device_type = rng.choice(DEVICE_TYPES, size=rows, p=[0.55, 0.35, 0.1])
    auth_method = rng.choice(AUTH_METHODS, size=rows, p=[0.6, 0.25, 0.15])
    mfa_used = rng.choice(MFA_USED, size=rows, p=[0.65, 0.35])
    asn = rng.integers(1000, 40000, size=rows).astype(str)

    login_failures_24h = rng.poisson(0.6, size=rows)
    unique_ips_24h = rng.poisson(1.2, size=rows)
    session_duration_s = rng.gamma(2.0, 600.0, size=rows)
    domain_similarity_score = rng.beta(2.0, 5.0, size=rows)
    suspicious_url_count = rng.poisson(0.4, size=rows)
    cert_age_days = rng.gamma(3.0, 40.0, size=rows)
    new_account_days = rng.integers(0, 60, size=rows)
    account_age_days = rng.integers(5, 3650, size=rows)
    hour_of_day = rng.integers(0, 24, size=rows)
    day_of_week = rng.integers(0, 7, size=rows)

    # Continuous, z-scored contributions rather than rare binary thresholds:
    # thresholded indicators (e.g. "suspicious_url_count > 1") are each true for
    # only a small slice of rows and rarely co-occur, so their combined signal
    # barely rises above the noise term below -- the resulting labels end up
    # almost unlearnable (~0.55-0.6 ROC-AUC ceiling even for a perfect model).
    # Scaled continuous signals give every row a graded, separable risk score.
    risk = (
        1.1 * _zscore(login_failures_24h)
        + 0.9 * _zscore(unique_ips_24h)
        + 1.4 * _zscore(domain_similarity_score)
        + 1.1 * _zscore(suspicious_url_count)
        - 1.0 * _zscore(cert_age_days)
        - 0.9 * _zscore(new_account_days)
        + 0.6 * (mfa_used == "no").astype(float)
        + 0.4 * (protocol == "pptp").astype(float)
    )

    logits = -3.5 + risk + rng.normal(0, 0.5, size=rows)
    prob = _sigmoid(logits)
    is_phishing = (rng.random(size=rows) < prob).astype(int)

    data = {
        "country": country,
        "region": region,
        "asn": asn,
        "vpn_provider": vpn_provider,
        "protocol": protocol,
        "device_type": device_type,
        "auth_method": auth_method,
        "mfa_used": mfa_used,
        "login_failures_24h": login_failures_24h,
        "unique_ips_24h": unique_ips_24h,
        "session_duration_s": session_duration_s,
        "domain_similarity_score": domain_similarity_score,
        "suspicious_url_count": suspicious_url_count,
        "cert_age_days": cert_age_days,
        "new_account_days": new_account_days,
        "account_age_days": account_age_days,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_phishing": is_phishing,
    }

    df = pd.DataFrame(data, columns=ALL_COLUMNS)
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic VPN phishing dataset")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--rows", type=int, default=50000, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic(args.rows, seed=args.seed)
    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
