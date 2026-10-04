import numpy as np
import pytest

from phishurl.metrics import (
    business_metrics,
    classification_metrics,
    precision_at_base_rate,
    threshold_for_fpr,
)


def test_classification_metrics_on_known_confusion_matrix():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1]
    m = classification_metrics(y, p, threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 4, "fp": 1, "fn": 1, "tp": 2}
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["accuracy"] == pytest.approx(6 / 8)
    assert m["false_positive_rate"] == pytest.approx(0.2)
    assert 0 < m["log_loss"] and 0 < m["brier"] < 1


def test_threshold_for_fpr_never_exceeds_target():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1000), np.ones(300)].astype(int)
    p = np.r_[rng.random(1000) * 0.8, 0.5 + rng.random(300) * 0.5]
    for target in (0.0, 0.01, 0.05, 0.2):
        thr = threshold_for_fpr(y, p, target)
        fpr = ((p >= thr) & (y == 0)).sum() / (y == 0).sum()
        assert fpr <= target + 1e-12


def test_precision_at_base_rate():
    # 80% recall, 1% FPR, 1% of traffic is phishing -> 0.008 / (0.008 + 0.0099)
    assert precision_at_base_rate(0.8, 0.01, 0.01) == pytest.approx(0.008 / 0.0179)
    assert precision_at_base_rate(0.0, 0.0, 0.01) == 0.0


def test_business_metrics_units():
    b = business_metrics(live_recall=0.75, legit_fpr=0.002)
    assert b["phishing_caught_per_100_live_sites"] == 75.0
    assert b["false_alarms_per_10k_legit_sites"] == 20.0
    assert set(b["precision_in_production"]) == {"phishing_is_1.0%_of_visits", "phishing_is_0.1%_of_visits"}
