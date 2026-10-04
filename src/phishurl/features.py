"""
Turn a raw URL into model features.

Only the registrable domain is used (e.g. ``paypal-verify.com`` or
``caseid42.firebaseapp.com``). The scheme, a leading ``www.``, extra
subdomains, the path and the query string are all dropped on purpose:
in PhiUSIIL every legitimate URL is a bare ``https://www.<domain>``
homepage, so any of those parts would let a model separate the classes
by URL *format* instead of learning what phishing domains look like.
See ``audit.py`` and the README for the numbers behind that decision.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from dataclasses import dataclass
from urllib.parse import urlsplit

import pandas as pd
import tldextract

# Offline extractor: uses the Public Suffix List snapshot bundled with the
# pinned tldextract version, so training and serving split domains the same
# way and nothing is fetched at runtime. Private suffixes (blogspot.com,
# pages.dev, firebaseapp.com, ...) are kept so hosting-platform subdomains
# count as their own registrable domain.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)

# Brands and lure words that show up constantly in phishing domains. Fixed
# up front from public phishing reports, not tuned on the evaluation data.
BRAND_TOKENS = (
    "paypal",
    "apple",
    "icloud",
    "microsoft",
    "office",
    "outlook",
    "onedrive",
    "netflix",
    "amazon",
    "facebook",
    "instagram",
    "whatsapp",
    "google",
    "gmail",
    "chase",
    "wellsfargo",
    "dhl",
    "usps",
    "fedex",
    "adobe",
    "dropbox",
    "docusign",
    "linkedin",
    "coinbase",
    "binance",
    "metamask",
    "steam",
)
LURE_TOKENS = (
    "login",
    "signin",
    "logon",
    "verify",
    "verification",
    "secure",
    "account",
    "update",
    "support",
    "wallet",
    "auth",
    "recover",
    "billing",
    "invoice",
    "confirm",
    "unlock",
    "suspend",
    "webmail",
)

NUMERIC_FEATURES = [
    "name_len",
    "name_digits",
    "name_digit_ratio",
    "name_hyphens",
    "name_entropy",
    "name_vowel_ratio",
    "name_max_consonant_run",
    "name_max_digit_run",
    "brand_token_count",
    "lure_token_count",
    "suffix_labels",
    "is_private_suffix",
    "host_is_suffix",
    "is_ip",
    "is_punycode",
]
CATEGORICAL_FEATURES = ["tld"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

_VOWELS = set("aeiou")
_WWW = re.compile(r"^www\d*\.")


@dataclass(frozen=True)
class DomainParts:
    host: str
    name: str
    suffix: str
    registrable: str
    is_private: bool
    is_ip: bool


def canonical_host(url: str) -> str:
    """Lower-cased hostname without scheme, credentials, port or leading ``www.``."""
    url = str(url).strip()
    if "://" not in url:
        url = "http://" + url
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        host = ""
    host = host.strip(".").lower()
    return _WWW.sub("", host)


def split_domain(url: str) -> DomainParts:
    host = canonical_host(url)
    try:
        ipaddress.ip_address(host)
        return DomainParts(host, host, "", host, False, True)
    except ValueError:
        pass
    ext = _EXTRACT(host)
    if ext.domain:
        registrable = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    else:
        registrable = ext.suffix or host
    return DomainParts(host, ext.domain, ext.suffix, registrable, bool(ext.is_private), False)


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _max_run(text: str, predicate) -> int:
    best = run = 0
    for ch in text:
        run = run + 1 if predicate(ch) else 0
        best = max(best, run)
    return best


def domain_features(url: str) -> dict:
    parts = split_domain(url)
    name = "" if parts.is_ip else parts.name
    letters = [c for c in name if c.isalpha()]
    digits = sum(c.isdigit() for c in name)
    tld = "<ip>" if parts.is_ip else (parts.suffix.rsplit(".", 1)[-1] if parts.suffix else "<none>")
    return {
        "name_len": len(name),
        "name_digits": digits,
        "name_digit_ratio": digits / len(name) if name else 0.0,
        "name_hyphens": name.count("-"),
        "name_entropy": _entropy(name),
        "name_vowel_ratio": sum(c in _VOWELS for c in letters) / len(letters) if letters else 0.0,
        "name_max_consonant_run": _max_run(name, lambda c: c.isalpha() and c not in _VOWELS),
        "name_max_digit_run": _max_run(name, str.isdigit),
        "brand_token_count": sum(tok in name for tok in BRAND_TOKENS),
        "lure_token_count": sum(tok in name for tok in LURE_TOKENS),
        "suffix_labels": parts.suffix.count(".") + 1 if parts.suffix else 0,
        "is_private_suffix": int(parts.is_private),
        "host_is_suffix": int(not parts.is_ip and not parts.name),
        "is_ip": int(parts.is_ip),
        "is_punycode": int("xn--" in parts.host),
        "tld": tld,
    }


def featurize(urls) -> pd.DataFrame:
    """Feature frame for an iterable of URLs, columns ordered as FEATURE_COLUMNS."""
    rows = [domain_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def registrable_domain(url: str) -> str:
    return split_domain(url).registrable
