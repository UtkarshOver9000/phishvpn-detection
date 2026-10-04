"""
Classification and business metrics, all computed from predictions on real data.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-7, 1 - 1e-7)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "rows": int(len(y_true)),
        "accuracy": round(accuracy_score(y_true, pred), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "log_loss": round(log_loss(y_true, proba, labels=[0, 1]), 4),
        "brier": round(brier_score_loss(y_true, proba), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def threshold_for_fpr(y_true, proba, max_fpr: float) -> float:
    """Lowest threshold whose false-positive rate on (y_true, proba) is <= max_fpr."""
    y_true = np.asarray(y_true).astype(int)
    legit_scores = np.sort(np.asarray(proba, dtype=float)[y_true == 0])
    if len(legit_scores) == 0:
        return 0.5
    allowed = int(np.floor(max_fpr * len(legit_scores)))
    # Flag only scores strictly above the (allowed+1)-th highest legitimate score.
    cutoff = legit_scores[-(allowed + 1)]
    return float(np.nextafter(cutoff, 1.0))


def precision_at_base_rate(recall: float, fpr: float, base_rate: float) -> float:
    """Share of flagged sites that are really phishing when phishing is `base_rate` of traffic."""
    caught = recall * base_rate
    false_alarms = fpr * (1 - base_rate)
    return caught / (caught + false_alarms) if (caught + false_alarms) else 0.0


def business_metrics(live_recall: float, legit_fpr: float, base_rates=(0.01, 0.001)) -> dict:
    """Operational numbers a security team would ask for, from live-data recall and FPR."""
    return {
        "phishing_caught_per_100_live_sites": round(100 * live_recall, 1),
        "false_alarms_per_10k_legit_sites": round(10_000 * legit_fpr, 1),
        "precision_in_production": {
            f"phishing_is_{base_rate:.1%}_of_visits": round(
                precision_at_base_rate(live_recall, legit_fpr, base_rate), 4
            )
            for base_rate in base_rates
        },
    }
