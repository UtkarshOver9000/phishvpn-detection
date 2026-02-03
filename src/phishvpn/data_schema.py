"""
Core data schema for VPN phishing detection.

This schema is designed to be practical for global-scale VPN telemetry,
covering geo/network, session behavior, and security signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


TARGET_COLUMN = "is_phishing"

NUMERIC_COLUMNS: List[str] = [
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

CATEGORICAL_COLUMNS: List[str] = [
    "country",
    "region",
    "asn",
    "vpn_provider",
    "protocol",
    "device_type",
    "auth_method",
    "mfa_used",
]

ALL_COLUMNS: List[str] = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + [TARGET_COLUMN]


@dataclass(frozen=True)
class Schema:
    categorical: List[str]
    numeric: List[str]
    target: str


SCHEMA = Schema(
    categorical=CATEGORICAL_COLUMNS,
    numeric=NUMERIC_COLUMNS,
    target=TARGET_COLUMN,
)
