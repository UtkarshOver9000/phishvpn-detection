"""
Train, evaluate and export the phishing-domain model on real data.

    python -m phishurl.train --data-dir data

Two training setups are compared on the same held-out data:

* ``phiusiil_only``: legitimate examples come only from PhiUSIIL.
* ``phiusiil_plus_tranco`` (shipped): 60% of the current Tranco top-1M list
  is added as extra legitimate examples. PhiUSIIL's legitimate domains are a
  narrow sample, and a model trained on them alone flags far too many real
  popular sites.

Every model is scored on PhiUSIIL test domains, on held-out Tranco domains,
and on today's OpenPhish feed, which no model ever trains on. The shipped
model goes to ``src/phishurl/artifacts/``; metrics and figures go to
``reports/``.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import plots
from .audit import run_audit
from .data import dedupe_by_domain, load_openphish, load_phiusiil, load_tranco, split_by_domain, split_tranco
from .explain import reference_ranges
from .features import FEATURE_COLUMNS, featurize
from .metrics import business_metrics, classification_metrics, threshold_for_fpr
from .models import train_boosting, train_logistic, train_mlp

PACKAGE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PACKAGE_DIR / "artifacts"
# Operating points, chosen on validation data. A browser-level phishing warning
# must rarely fire on real sites, so the shipped model uses 1 false alarm per
# 1,000 legitimate domains; 1 per 100 is reported for comparison.
OPERATING_POINTS = {"fpr_0.1%": 0.001, "fpr_1%": 0.01}
SHIPPED_POINT = "fpr_0.1%"
SPLITS = ("train", "val", "test")


def _evaluate(model, threshold: float, sets: dict) -> dict:
    out = {}
    for name, (x, y) in sets.items():
        proba = model.predict_proba(x)[:, 1]
        if y is None:  # OpenPhish: every row is phishing
            out[name] = {"rows": int(len(proba)), "recall": round(float((proba >= threshold).mean()), 4)}
        elif y.sum() == 0:  # Tranco: every row is legitimate
            out[name] = {"rows": int(len(proba)), "false_positive_rate": round(float((proba >= threshold).mean()), 4)}
        else:
            out[name] = classification_metrics(y, proba, threshold)
    return out


def run(data_dir: Path, reports_dir: Path, seed: int = 7) -> dict:
    t0 = time.time()
    phi, dedupe_stats = dedupe_by_domain(load_phiusiil(data_dir))
    phi_splits = split_by_domain(phi, seed=seed)
    tranco_all, tranco_meta = load_tranco(data_dir)
    tranco, tranco_stats = split_tranco(tranco_all, exclude=set(phi["domain"]), seed=seed)
    openphish, openphish_meta = load_openphish(data_dir)

    phi_x = {s: featurize(getattr(phi_splits, s)["url"]) for s in SPLITS}
    phi_y = {s: getattr(phi_splits, s)["is_phishing"].to_numpy() for s in SPLITS}
    tr_x = {s: featurize(tranco[s]["url"]) for s in SPLITS}
    tr_y = {s: np.zeros(len(tranco[s]), dtype=int) for s in SPLITS}
    live_x = featurize(openphish["url"])

    def both(split):
        return pd.concat([phi_x[split], tr_x[split]], ignore_index=True), np.concatenate([phi_y[split], tr_y[split]])

    eval_sets = {
        "phiusiil_test": (phi_x["test"], phi_y["test"]),
        "combined_test": both("test"),
        "tranco_test_legit": (tr_x["test"], tr_y["test"]),
        "openphish_live": (live_x, None),
    }

    # Setup 1 (ablation): PhiUSIIL only.
    ablation, ablation_hist = train_boosting(phi_x["train"], phi_y["train"], phi_x["val"], phi_y["val"], seed=seed)
    ablation_val = ablation.predict_proba(phi_x["val"])[:, 1]
    ablation_thresholds = {p: threshold_for_fpr(phi_y["val"], ablation_val, f) for p, f in OPERATING_POINTS.items()}

    # Setup 2 (shipped): PhiUSIIL + Tranco legitimate domains.
    x_train, y_train = both("train")
    x_val, y_val = both("val")
    models, histories = {}, {}
    models["logistic_regression"], histories["logistic_regression"] = train_logistic(x_train, y_train, x_val, y_val)
    models["gradient_boosting"], histories["gradient_boosting"] = train_boosting(
        x_train, y_train, x_val, y_val, seed=seed
    )
    models["mlp"], histories["mlp"] = train_mlp(x_train, y_train, x_val, y_val, seed=seed)

    results, thresholds, test_proba = {}, {}, {}
    for name, model in models.items():
        val_proba = model.predict_proba(x_val)[:, 1]
        thresholds[name] = {p: threshold_for_fpr(y_val, val_proba, f) for p, f in OPERATING_POINTS.items()}
        test_proba[name] = model.predict_proba(eval_sets["combined_test"][0])[:, 1]
        results[name] = {
            "thresholds": {p: round(t, 4) for p, t in thresholds[name].items()},
            **{f"at_{p}": _evaluate(model, t, eval_sets) for p, t in thresholds[name].items()},
            "at_0.5": _evaluate(model, 0.5, eval_sets),
            "train_at_shipped_point": classification_metrics(
                y_train, model.predict_proba(x_train)[:, 1], thresholds[name][SHIPPED_POINT]
            ),
        }

    shipped = "gradient_boosting"
    model, threshold = models[shipped], thresholds[shipped][SHIPPED_POINT]
    shipped_eval = results[shipped][f"at_{SHIPPED_POINT}"]
    live_recall = shipped_eval["openphish_live"]["recall"]
    legit_fpr = shipped_eval["tranco_test_legit"]["false_positive_rate"]
    train_domains = set(phi_splits.train["domain"])

    report = {
        "datasets": {
            "phiusiil": {
                "name": "PhiUSIIL Phishing URL Dataset (UCI #967)",
                "source": "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset",
                **dedupe_stats,
                "split": phi_splits.stats,
            },
            "tranco": {**tranco_meta, "split": tranco_stats},
            "openphish": {
                **openphish_meta,
                "domains_also_in_phiusiil_train": int(openphish["domain"].isin(train_domains).sum()),
            },
        },
        "features": FEATURE_COLUMNS,
        "operating_points": OPERATING_POINTS,
        "shipped_operating_point": SHIPPED_POINT,
        "ablation_phiusiil_only": {
            "best_round": ablation_hist["best_round"],
            "thresholds": {p: round(t, 4) for p, t in ablation_thresholds.items()},
            **{f"at_{p}": _evaluate(ablation, t, eval_sets) for p, t in ablation_thresholds.items()},
        },
        "training": histories,
        "results": results,
        "shipped_model": shipped,
        "shipped_threshold": round(threshold, 4),
        "business": business_metrics(live_recall, legit_fpr),
        "shortcut_audit": run_audit(data_dir, seed=seed),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }

    figures = reports_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plots.loss_curves(histories["gradient_boosting"], histories["mlp"], figures / "loss_curves.png")
    plots.roc_pr(eval_sets["combined_test"][1], test_proba, figures / "roc_pr.png")
    plots.confusion(
        shipped_eval["combined_test"]["confusion_matrix"],
        f"Gradient boosting, test set (threshold {threshold:.3f})",
        figures / "confusion_matrix.png",
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "model.joblib", compress=3)
    card = {
        "model": shipped,
        "threshold": round(threshold, 4),
        "features": FEATURE_COLUMNS,
        "legit_reference_ranges": reference_ranges(x_train[y_train == 0]),
        "test_metrics": shipped_eval["combined_test"],
        "live_evaluation": {
            "openphish_recall": live_recall,
            "openphish_fetched_at": openphish_meta["fetched_at"],
            "tranco_false_positive_rate": legit_fpr,
            "tranco_list_id": tranco_meta["list_id"],
        },
        "business": report["business"],
        "scikit_learn": sklearn.__version__,
    }
    (ARTIFACT_DIR / "model_card.json").write_text(json.dumps(card, indent=2))

    report["runtime_seconds"] = round(time.time() - t0, 1)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the phishing-domain model on real data and evaluate it live")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    report = run(args.data_dir, args.reports_dir, seed=args.seed)
    summary = {
        "shipped": report["results"][report["shipped_model"]][f"at_{SHIPPED_POINT}"],
        "ablation_phiusiil_only": report["ablation_phiusiil_only"][f"at_{SHIPPED_POINT}"],
        "business": report["business"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
