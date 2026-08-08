"""
End-to-end tests for the train / evaluate / infer CLI entrypoints,
run against a small synthetic dataset in a temp directory.
"""

import json

from phishvpn import evaluate as evaluate_cli
from phishvpn import infer as infer_cli
from phishvpn import train as train_cli
from phishvpn.synthetic_data import generate_synthetic


def _write_dataset(tmp_path, rows=800, seed=5):
    data_path = tmp_path / "sessions.csv"
    generate_synthetic(rows, seed=seed).to_csv(data_path, index=False)
    return data_path


def test_train_cli_writes_model_and_metrics(tmp_path, monkeypatch):
    data_path = _write_dataset(tmp_path)
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--data", str(data_path),
            "--model-out", str(model_path),
            "--metrics-out", str(metrics_path),
            "--seed", "5",
        ],
    )
    train_cli.main()

    assert model_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_evaluate_cli_writes_metrics(tmp_path, monkeypatch):
    data_path = _write_dataset(tmp_path)
    model_path = tmp_path / "model.joblib"
    train_metrics_path = tmp_path / "train_metrics.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--data", str(data_path),
            "--model-out", str(model_path),
            "--metrics-out", str(train_metrics_path),
            "--seed", "5",
        ],
    )
    train_cli.main()

    eval_out = tmp_path / "eval.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--data", str(data_path),
            "--model", str(model_path),
            "--out", str(eval_out),
        ],
    )
    evaluate_cli.main()

    metrics = json.loads(eval_out.read_text())
    assert "roc_auc" in metrics


def test_infer_cli_writes_predictions(tmp_path, monkeypatch):
    data_path = _write_dataset(tmp_path)
    model_path = tmp_path / "model.joblib"
    train_metrics_path = tmp_path / "train_metrics.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--data", str(data_path),
            "--model-out", str(model_path),
            "--metrics-out", str(train_metrics_path),
            "--seed", "5",
        ],
    )
    train_cli.main()

    preds_out = tmp_path / "preds.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "infer",
            "--data", str(data_path),
            "--model", str(model_path),
            "--out", str(preds_out),
        ],
    )
    infer_cli.main()

    import pandas as pd

    preds = pd.read_csv(preds_out)
    assert "phishing_risk_score" in preds.columns
    assert "phishing_pred" in preds.columns
    assert preds["phishing_pred"].isin([0, 1]).all()
    assert ((preds["phishing_risk_score"] >= 0) & (preds["phishing_risk_score"] <= 1)).all()
