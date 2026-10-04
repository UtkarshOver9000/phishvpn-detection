"""
Download the three real datasets used for training and evaluation.

    python scripts/download_data.py            # into ./data

* PhiUSIIL Phishing URL Dataset, UCI ML Repository #967 (CC BY 4.0), ~15 MB zip
* Tranco top-1M list, https://tranco-list.eu (current daily list), ~10 MB zip
* OpenPhish community feed, https://openphish.com/feed.txt (live, changes hourly)

The OpenPhish feed contains live malicious URLs. It is only read as text and
is never committed to the repository (``data/`` is gitignored).
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {
    "phiusiil.zip": "https://archive.ics.uci.edu/static/public/967/phiusiil+phishing+url+dataset.zip",
    "tranco-top-1m.csv.zip": "https://tranco-list.eu/download/daily/top-1m.csv.zip",
    "openphish-feed.txt": "https://openphish.com/feed.txt",
}
TRANCO_ID_URL = "https://tranco-list.eu/top-1m-id"
USER_AGENT = "phishing-url-detection-research/2.0"


def fetch(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    for name, url in SOURCES.items():
        dest = args.data_dir / name
        print(f"downloading {url}")
        fetch(url, dest)
        print(f"  {dest} ({dest.stat().st_size:,} bytes, sha256 {sha256(dest)[:16]}...)")

    with zipfile.ZipFile(args.data_dir / "phiusiil.zip") as zf:
        zf.extractall(args.data_dir)
    fetch(TRANCO_ID_URL, args.data_dir / "tranco-list-id.txt")
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (args.data_dir / "openphish-fetched-at.txt").write_text(stamp + "\n")
    print(f"Tranco list id: {(args.data_dir / 'tranco-list-id.txt').read_text().strip()}; OpenPhish fetched {stamp}")


if __name__ == "__main__":
    main()
