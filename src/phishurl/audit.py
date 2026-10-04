"""
Shortcut audit for PhiUSIIL.

Measures how well trivial signals separate the classes. If a one-line rule
scores ~99%, a model's 99% means nothing. These numbers are why the model
in this repo only looks at the registrable domain.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import PHIUSIIL_CSV


def run_audit(data_dir: Path, seed: int = 7) -> dict:
    df = pd.read_csv(Path(data_dir) / PHIUSIIL_CSV, encoding="utf-8-sig")
    y = (df["label"] == 0).astype(int).to_numpy()  # 1 = phishing
    url = df["URL"].astype(str)

    legit = df["label"] == 1
    format_rule_pred = (~url.str.startswith("https://www.")).astype(int).to_numpy()

    numeric = df.select_dtypes("number").drop(columns=["label"])
    single_auc = {c: roc_auc_score(y, numeric[c]) for c in numeric.columns}
    single_auc = {c: max(a, 1 - a) for c, a in single_auc.items()}
    top = sorted(single_auc.items(), key=lambda kv: -kv[1])[:5]

    x_tr, x_te, y_tr, y_te = train_test_split(numeric, y, test_size=0.2, random_state=seed, stratify=y)
    all_features = HistGradientBoostingClassifier(random_state=seed).fit(x_tr, y_tr)

    return {
        "legitimate_urls_starting_https_www": round(float(url[legit].str.startswith("https://www.").mean()), 4),
        "phishing_urls_starting_https_www": round(float(url[~legit].str.startswith("https://www.").mean()), 4),
        "legitimate_urls_with_a_path": round(
            float(url[legit].str.replace(r"^https?://", "", regex=True).str.contains("/.", regex=True).mean()), 4
        ),
        "rule_not_https_www_is_phishing_accuracy": round(float(accuracy_score(y, format_rule_pred)), 4),
        "best_single_feature_auc": {name: round(float(auc), 4) for name, auc in top},
        "numeric_features_used": int(numeric.shape[1]),
        "all_numeric_features_test_accuracy": round(float(accuracy_score(y_te, all_features.predict(x_te))), 4),
        "all_numeric_features_test_auc": round(
            float(roc_auc_score(y_te, all_features.predict_proba(x_te)[:, 1])), 4
        ),
        "rows": int(len(df)),
        "note": (
            "Every legitimate URL is a bare https://www.<domain> homepage, so URL format alone nearly "
            "separates the classes, and single precomputed features reach near-perfect AUC on their own, "
            "so a near-100% score on the full feature set says little about real-world performance."
        ),
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Shortcut audit of the PhiUSIIL dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    np.set_printoptions(precision=4)
    print(json.dumps(run_audit(args.data_dir), indent=2))
