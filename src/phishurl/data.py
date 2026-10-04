"""
Loaders for the three real datasets this project uses.

* PhiUSIIL (UCI #967): 235,795 labelled URLs, used for training and the
  held-out test set.
* OpenPhish community feed: phishing URLs that are live right now, used only
  for evaluation.
* Tranco top-1M: the current list of popular registrable domains, used only
  to measure false alarms on legitimate sites.

``scripts/download_data.py`` fetches all three into ``data/``.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import registrable_domain

PHIUSIIL_CSV = "PhiUSIIL_Phishing_URL_Dataset.csv"
OPENPHISH_TXT = "openphish-feed.txt"
TRANCO_ZIP = "tranco-top-1m.csv.zip"


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    stats: dict


def load_phiusiil(data_dir: Path) -> pd.DataFrame:
    """URL + is_phishing label. PhiUSIIL encodes legitimate as 1, so it is flipped here."""
    df = pd.read_csv(Path(data_dir) / PHIUSIIL_CSV, usecols=["URL", "label"], encoding="utf-8-sig")
    return pd.DataFrame({"url": df["URL"].astype(str), "is_phishing": (df["label"] == 0).astype(int)})


def dedupe_by_domain(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """One row per registrable domain, dropping domains that carry both labels.

    The model only sees the registrable domain, so several URLs on one domain
    would be identical training rows, and the same domain must never sit on
    both sides of the train/test split.
    """
    df = df.copy()
    df["domain"] = df["url"].map(registrable_domain)
    df = df[df["domain"] != ""]
    labels_per_domain = df.groupby("domain")["is_phishing"].nunique()
    conflicting = set(labels_per_domain[labels_per_domain > 1].index)
    clean = df[~df["domain"].isin(conflicting)].drop_duplicates("domain").reset_index(drop=True)
    stats = {
        "raw_rows": int(len(df)),
        "conflicting_domains_dropped": len(conflicting),
        "unique_domains": int(len(clean)),
        "phishing_domains": int(clean["is_phishing"].sum()),
        "legitimate_domains": int((clean["is_phishing"] == 0).sum()),
    }
    return clean, stats


def split_by_domain(df: pd.DataFrame, seed: int = 7, val: float = 0.15, test: float = 0.15) -> Splits:
    """Stratified 70/15/15 split over unique domains (rows are already one per domain)."""
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for _, group in df.groupby("is_phishing"):
        idx = rng.permutation(group.index.to_numpy())
        n_test = int(round(len(idx) * test))
        n_val = int(round(len(idx) * val))
        parts["test"].append(idx[:n_test])
        parts["val"].append(idx[n_test : n_test + n_val])
        parts["train"].append(idx[n_test + n_val :])
    frames = {
        k: df.loc[np.concatenate(v)].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        for k, v in parts.items()
    }
    stats = {k: {"rows": int(len(v)), "phishing": int(v["is_phishing"].sum())} for k, v in frames.items()}
    return Splits(frames["train"], frames["val"], frames["test"], stats)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_openphish(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    path = Path(data_dir) / OPENPHISH_TXT
    urls = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame({"url": urls})
    df["domain"] = df["url"].map(registrable_domain)
    df = df[df["domain"] != ""].drop_duplicates("domain").reset_index(drop=True)
    fetched = Path(data_dir) / "openphish-fetched-at.txt"
    meta = {
        "source": "https://openphish.com/feed.txt",
        "fetched_at": fetched.read_text().strip() if fetched.exists() else None,
        "urls_in_feed": len(urls),
        "unique_domains": int(len(df)),
        "sha256": _sha256(path),
    }
    return df, meta


def split_tranco(tranco: pd.DataFrame, exclude: set, seed: int = 7) -> tuple[dict, dict]:
    """60/20/20 random split of Tranco domains not already present in PhiUSIIL."""
    df = tranco[~tranco["domain"].isin(exclude)].copy()
    df["url"] = df["domain"]
    df["is_phishing"] = 0
    idx = np.random.default_rng(seed).permutation(len(df))
    n_test = n_val = int(round(len(df) * 0.2))
    parts = {
        "test": df.iloc[idx[:n_test]],
        "val": df.iloc[idx[n_test : n_test + n_val]],
        "train": df.iloc[idx[n_test + n_val :]],
    }
    parts = {k: v.reset_index(drop=True) for k, v in parts.items()}
    stats = {"overlap_with_phiusiil_dropped": int(len(tranco) - len(df))} | {k: int(len(v)) for k, v in parts.items()}
    return parts, stats


def load_tranco(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    path = Path(data_dir) / TRANCO_ZIP
    with zipfile.ZipFile(path) as zf, zf.open("top-1m.csv") as fh:
        df = pd.read_csv(fh, header=None, names=["rank", "domain"])
    list_id_file = Path(data_dir) / "tranco-list-id.txt"
    meta = {
        "source": "https://tranco-list.eu/",
        "list_id": list_id_file.read_text().strip() if list_id_file.exists() else None,
        "domains": int(len(df)),
        "sha256": _sha256(path),
    }
    return df, meta
