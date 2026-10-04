import numpy as np
import pytest

from phishurl.explain import reasons, reference_ranges
from phishurl.features import featurize
from phishurl.models import train_boosting, train_logistic, train_mlp

LEGIT = [
    "google.com",
    "wikipedia.org",
    "bbc.co.uk",
    "github.com",
    "python.org",
    "mozilla.org",
    "apache.org",
    "stackoverflow.com",
    "nytimes.com",
    "reddit.com",
    "amazon.in",
    "flipkart.com",
    "irctc.co.in",
    "nasa.gov",
]
PHISH = [
    "secure-paypal-login-verify.com",
    "x9k2-account-update.top",
    "appleid-unlock-support.xyz",
    "caseid1008934573.firebaseapp.com",
    "m1crosoft-office-verify.info",
    "wallet-recover-metamask.click",
    "dhl-parcel-confirm-2291.shop",
    "signin-amazon-billing.live",
    "198.51.100.23",
    "netflix-billing-update.cc",
]


@pytest.fixture(scope="module")
def data():
    urls = LEGIT * 4 + PHISH * 4
    y = np.r_[np.zeros(len(LEGIT) * 4), np.ones(len(PHISH) * 4)].astype(int)
    return featurize(urls), y


def test_logistic_regression_reports_losses(data):
    x, y = data
    model, hist = train_logistic(x, y, x, y)
    assert hist["train_log_loss"] > 0 and hist["solver_iterations"] >= 1
    assert model.predict_proba(x).shape == (len(y), 2)


def test_boosting_keeps_best_validation_round(data):
    x, y = data
    model, hist = train_boosting(x, y, x, y, max_rounds=30)
    assert len(hist["train_log_loss_per_round"]) == 30 == len(hist["val_log_loss_per_round"])
    best = hist["best_round"]
    assert 1 <= best <= 30
    assert hist["val_log_loss_per_round"][best - 1] == min(hist["val_log_loss_per_round"])
    assert model.named_steps["clf"].n_iter_ == best


def test_mlp_records_every_epoch(data):
    x, y = data
    model, hist = train_mlp(x, y, x, y, max_epochs=5, patience=10)
    assert hist["epochs_run"] == 5
    assert len(hist["train_log_loss_per_epoch"]) == len(hist["val_accuracy_per_epoch"]) == 5
    assert 1 <= hist["best_epoch"] <= 5
    proba = model.predict_proba(x)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()


def test_reasons_flag_values_outside_legit_range():
    x = featurize(LEGIT)
    ranges = reference_ranges(x)
    suspicious = featurize(["secure-paypal-login-verify-account-77812.com"]).iloc[0].to_dict()
    out = reasons(suspicious, ranges)
    assert out and any("domain name length" in r for r in out)
    assert reasons(featurize(["google.com"]).iloc[0].to_dict(), ranges) == []
