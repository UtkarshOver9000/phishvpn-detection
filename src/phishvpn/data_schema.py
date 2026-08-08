"""
Core data schema for VPN phishing detection.

This schema is designed to be practical for global-scale VPN telemetry,
covering geo/network, session behavior, and security signals.
"""

from __future__ import annotations

from dataclasses import dataclass

TARGET_COLUMN = "is_phishing"

NUMERIC_COLUMNS: list[str] = [
    "login_failures_24h",
    "unique_ips_24h",
    "session_duration_s",
    "domain_similarity_score",
    "suspicious_url_count",
    "cert_age_days",
    "new_account_days",
    "account_age_days",
    "hour_of_day",
    "day_of_week",
]

CATEGORICAL_COLUMNS: list[str] = [
    "country",
    "region",
    "vpn_provider",
    "protocol",
    "device_type",
    "auth_method",
    "mfa_used",
]

# Recorded for telemetry/joins but deliberately excluded from CATEGORICAL_COLUMNS:
# ASN is a near-unique identifier per session (tens of thousands of distinct
# values), so one-hot encoding it as a model feature lets a linear model
# memorize training rows instead of generalizing -- it consistently tanked
# held-out ROC-AUC to ~0.55-0.58 in testing, versus ~0.92 without it.
IDENTIFIER_COLUMNS: list[str] = ["asn"]

ALL_COLUMNS: list[str] = IDENTIFIER_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [TARGET_COLUMN]


@dataclass(frozen=True)
class Schema:
    categorical: list[str]
    numeric: list[str]
    target: str


SCHEMA = Schema(
    categorical=CATEGORICAL_COLUMNS,
    numeric=NUMERIC_COLUMNS,
    target=TARGET_COLUMN,
)
