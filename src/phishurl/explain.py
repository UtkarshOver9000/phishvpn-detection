"""
Plain-language reasons for a score.

Each numeric feature of the scored domain is compared with the range seen on
legitimate training domains (5th-95th percentile). Values outside that range
are reported. This describes what is unusual about the domain; it is not an
exact attribution of the model's output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import NUMERIC_FEATURES

LABELS = {
    "name_len": "domain name length",
    "name_digits": "digits in the domain name",
    "name_digit_ratio": "share of digits in the domain name",
    "name_hyphens": "hyphens in the domain name",
    "name_entropy": "randomness (entropy) of the domain name",
    "name_vowel_ratio": "share of vowels in the domain name",
    "name_max_consonant_run": "longest run of consonants",
    "name_max_digit_run": "longest run of digits",
    "brand_token_count": "brand names inside the domain (e.g. 'paypal')",
    "lure_token_count": "lure words inside the domain (e.g. 'login', 'verify')",
    "suffix_labels": "parts in the domain suffix",
    "is_private_suffix": "hosted on a free/shared platform subdomain (e.g. *.pages.dev)",
    "host_is_suffix": "URL points at a shared hosting platform itself",
    "is_ip": "raw IP address instead of a domain",
    "is_punycode": "punycode (look-alike unicode) domain",
}


def reference_ranges(x_legit: pd.DataFrame) -> dict:
    return {
        col: {
            "p05": float(np.percentile(x_legit[col], 5)),
            "p95": float(np.percentile(x_legit[col], 95)),
            "mean": float(x_legit[col].mean()),
        }
        for col in NUMERIC_FEATURES
    }


def reasons(features: dict, ranges: dict, limit: int = 4) -> list[str]:
    out = []
    for col in NUMERIC_FEATURES:
        value, ref = features[col], ranges[col]
        if value > ref["p95"]:
            direction = "above"
        elif value < ref["p05"]:
            direction = "below"
        else:
            continue
        shown = f"{value:.2f}" if isinstance(value, float) and not float(value).is_integer() else f"{int(value)}"
        if ref["p95"] == ref["p05"]:
            out.append(f"{LABELS[col]}: {shown} (legitimate domains are almost always {ref['p05']:g})")
        else:
            out.append(
                f"{LABELS[col]}: {shown}, {direction} the usual range for legitimate domains "
                f"({ref['p05']:.2f}-{ref['p95']:.2f})"
            )
    return out[:limit]
