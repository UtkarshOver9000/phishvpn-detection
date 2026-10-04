import io
import zipfile

import pandas as pd

from phishurl.data import (
    dedupe_by_domain,
    load_openphish,
    load_phiusiil,
    load_tranco,
    split_by_domain,
    split_tranco,
)


def _frame(rows):
    return pd.DataFrame(rows, columns=["url", "is_phishing"])


def test_phiusiil_label_is_flipped_to_is_phishing(tmp_path):
    (tmp_path / "PhiUSIIL_Phishing_URL_Dataset.csv").write_text(
        "﻿FILENAME,URL,label\na.txt,https://www.good.com,1\nb.txt,http://bad.example.net/x,0\n", encoding="utf-8"
    )
    df = load_phiusiil(tmp_path)
    assert df["is_phishing"].tolist() == [0, 1]


def test_dedupe_drops_conflicts_and_duplicates():
    df = _frame(
        [
            ("https://www.shop.com", 0),
            ("https://shop.com/a", 0),  # same domain, same label -> one row
            ("http://evil.com/a", 1),
            ("https://www.evil.com", 0),  # same domain, other label -> dropped
            ("http://x.blogspot.com", 1),
        ]
    )
    clean, stats = dedupe_by_domain(df)
    assert sorted(clean["domain"]) == ["shop.com", "x.blogspot.com"]
    assert stats["conflicting_domains_dropped"] == 1


def test_split_has_no_domain_overlap_and_keeps_class_balance():
    rows = [(f"https://legit{i}.com", 0) for i in range(200)] + [(f"http://phish{i}.xyz", 1) for i in range(100)]
    clean, _ = dedupe_by_domain(_frame(rows))
    splits = split_by_domain(clean, seed=1)
    sets = [set(s["domain"]) for s in (splits.train, splits.val, splits.test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sum(len(s) for s in sets) == 300
    assert splits.stats["test"]["phishing"] == 15 and splits.stats["test"]["rows"] == 45


def test_tranco_split_excludes_known_domains():
    tranco = pd.DataFrame({"rank": range(1, 11), "domain": [f"d{i}.com" for i in range(10)]})
    parts, stats = split_tranco(tranco, exclude={"d0.com", "d1.com"}, seed=0)
    all_domains = set().union(*(set(p["domain"]) for p in parts.values()))
    assert "d0.com" not in all_domains and len(all_domains) == 8
    assert stats["overlap_with_phiusiil_dropped"] == 2
    assert all((p["is_phishing"] == 0).all() for p in parts.values())


def test_live_loaders(tmp_path):
    (tmp_path / "openphish-feed.txt").write_text("http://a.evil.com/x\nhttp://b.evil.com/y\nhttp://other.net\n\n")
    (tmp_path / "openphish-fetched-at.txt").write_text("2026-10-04T04:29:44Z\n")
    df, meta = load_openphish(tmp_path)
    assert meta["urls_in_feed"] == 3 and meta["unique_domains"] == 2
    assert meta["fetched_at"] == "2026-10-04T04:29:44Z"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("top-1m.csv", "1,google.com\n2,facebook.com\n")
    (tmp_path / "tranco-top-1m.csv.zip").write_bytes(buf.getvalue())
    (tmp_path / "tranco-list-id.txt").write_text("ABCDE")
    tranco, tmeta = load_tranco(tmp_path)
    assert tranco["domain"].tolist() == ["google.com", "facebook.com"]
    assert tmeta["list_id"] == "ABCDE" and len(tmeta["sha256"]) == 64
